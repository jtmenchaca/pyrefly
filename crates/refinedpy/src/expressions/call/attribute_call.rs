//! `receiver.attr(...)` dispatch: module-qualified calls (`math.sqrt`,
//! `re.search`, `json.loads`, …), method calls on known receiver shapes
//! (strings, dicts, lists, tagged datetime/regex/bytes instances), and
//! the class-scoping helper a chained method call's receiver needs.

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_text_size::TextRange;

use crate::assignability;
use crate::builtin_models;
use crate::bytes_models;
use crate::collection_models;
use crate::env::Environment;
use crate::instances;
use crate::json_grammar;
use crate::math_models;
use crate::string_models;

use super::super::arithmetic::single_numeric_value;
use super::super::attribute::boolean_answer;
use super::super::attribute::dict_view_method_result;
use super::super::attribute::set_method_result;
use super::super::compare::exact_string_values;
use super::super::compare::single_pair_equal;
use super::super::datetime::date_isocalendar_value;
use super::super::datetime::date_isoformat_value;
use super::super::datetime::datetime_isoformat_value;
use super::super::datetime::date_isoweekday_value;
use super::super::datetime::date_toordinal_value;
use super::super::datetime::date_weekday_value;
use super::super::datetime::datetime_timestamp_value;
use super::super::datetime::strftime_iso_date_value;
use super::super::datetime::strptime2_parse;
use super::super::datetime::strptime2_scan_format;
use super::super::datetime::strptime_iso_date_value;
use super::super::datetime::is_datetime_datetime_attribute;
use super::super::evaluate_expression;
use super::super::fstring::code_points_to_string;
use super::super::json_re::attribute_chain_root_name;
use super::super::json_re::json_dumps_value;
use super::super::json_re::json_loads_value_space;
use super::super::json_re::json_scalar_literal_value;
use super::super::json_re::re_search_literal_value;
use super::super::json_re::MODELED_MODULE_NAMES;

/// The class table a chained method call's RECEIVER expression should
/// resolve its instance's class against, read fresh from the SPECIFIC
/// same-module def the receiver traces back to — never the caller's own
/// shared `environment.classes()`, which can hold only one entry per
/// bare class name and so cannot tell two sibling nested defs' own
/// same-named classes apart (`check.rs::local_class_table`'s own doc:
/// "a class nested inside a NESTED def... is collected too", flattened
/// into one map, first-scanned-wins on a spelling collision).
///
/// `receiver` is peeled one layer at a time: an `Attribute` reads
/// through to its own `.value` (`make_over_builder().type("x")`'s
/// receiver, for the `.size(1)` call, is `make_over_builder().type("x")`
/// itself — another Attribute call, not yet the root), a `Call` whose
/// callee is a bare `Name` naming a same-module def (`environment.
/// functions()`, which already carries every LOCAL nested def merged
/// over the module's own top-level ones — `check.rs::local_function_
/// table`'s own doc) is the root: that def's own body is rescanned for
/// its OWN top-level classes, mirroring `summaries::interpret_class_def`'s
/// exact synthetic-module construction (empty aliases/imports — a
/// body-local class's own field annotations reading a module-level
/// alias is a narrower, still-sound miss the same way that function's
/// own doc already accepts). `None` for every other receiver shape (an
/// ordinary bound name, a field read, a call to anything but a
/// same-module def) — the caller falls back to `environment.classes()`
/// unchanged.
pub(in super::super) fn receiver_def_local_classes(
    receiver: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<std::sync::Arc<std::collections::HashMap<String, instances::ClassModel>>> {
    match receiver {
        Expr::Attribute(attribute) => receiver_def_local_classes(&attribute.value, environment, kernel),
        // `make_over_builder().type("x")` (the `.size(1)` call's own
        // receiver) is ITSELF a Call whose callee is an Attribute, not
        // yet the root — peel through `.func`'s own receiver the same
        // way the `Expr::Attribute` arm above peels `.value`, so a chain
        // of any length still traces back to the one `Name` call that
        // started it.
        Expr::Call(call) if matches!(call.func.as_ref(), Expr::Attribute(_)) => {
            receiver_def_local_classes(call.func.as_ref(), environment, kernel)
        }
        Expr::Call(call) => {
            let Expr::Name(name) = call.func.as_ref() else {
                return None;
            };
            let def = environment.functions()?.def(name.id.as_str())?;
            let synthetic = ruff_python_ast::ModModule {
                node_index: ruff_python_ast::AtomicNodeIndex::NONE,
                range: TextRange::default(),
                body: def.body.iter().filter(|stmt| matches!(stmt, ruff_python_ast::Stmt::ClassDef(_))).cloned().collect(),
            };
            let empty_aliases = std::collections::HashMap::new();
            let empty_imports = crate::surface::surface_imports(&ruff_python_ast::ModModule {
                node_index: ruff_python_ast::AtomicNodeIndex::NONE,
                range: TextRange::default(),
                body: Vec::new().into(),
            });
            Some(std::sync::Arc::new(instances::class_table(&synthetic, &empty_aliases, &empty_imports, kernel)))
        }
        _ => None,
    }
}

/// Whether a known exact-string regex pattern contains NO metacharacter
/// `re.escape` would need to escape — library/re.html, `function::
/// escape(pattern)`: "Escape special characters in pattern... useful if
/// you want to match an arbitrary literal string that may have regular
/// expression metacharacters in it," and "The special characters are"
/// (re.html §"Regular Expression Syntax") `. ^ $ * + ? { } [ ] \ | ( )`.
/// A pattern containing none of those characters matches ITSELF and
/// only itself — `re.search`/`re.sub` over such a pattern reduce to a
/// plain substring test/replace, decidable without a regex engine
/// (`re_search_literal_value`/`re.sub`'s own call-site doc). `pattern`
/// must also be a known exact string; a non-string or unknown pattern
/// answers `false` (never metacharacter-free, since it is not even a
/// known literal).
pub(in super::super) fn is_literal_regex_pattern(pattern: &AbstractValue) -> bool {
    const REGEX_METACHARACTERS: &[char] = &['.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '\\', '|', '(', ')'];
    let Some(text) = exact_string_values(pattern).and_then(code_points_to_string) else {
        return false;
    };
    !text.chars().any(|c| REGEX_METACHARACTERS.contains(&c))
}

/// Rung 1 of the compiled-extension recognition ladder
/// (`python-c-extension-boundary.md`'s naming unit): whether `call` is a
/// call on an attribute chain rooted at an imported-but-unmodeled module
/// name — `torch.arange(5)`, `pandas.read_csv(...).head()`'s own
/// receiver — answering that root module's own name for the caller to
/// name in its decline sentence.
///
/// Recognized the same way every modeled-module arm in
/// `evaluate_attribute_call` recognizes ITS OWN module: the chain's root
/// is a bare `Name` that reads UNBOUND in `environment`
/// (`environment.read(name).is_none()` — the identical gate `math`/`re`/
/// `json`/etc. already apply, since an import that resolved to nothing
/// this checker tracks leaves the name unbound, `check.rs::
/// bind_or_forget_imported_name`'s own doc) AND is not itself one of the
/// `MODELED_MODULE_NAMES` this file already carries a model for (a
/// modeled module's own unmodeled FUNCTION — `math.frexp`, say — is a
/// different, narrower gap this naming unit does not claim; only a
/// module with NO model at all is named here). `None` for every other
/// call shape: a bare-name call, a method call on an evaluated (non-
/// module) receiver, or a call whose root name IS bound to a real
/// tracked value (shadowing the module the way every existing arm's own
/// gate already respects).
pub fn unmodeled_module_call_name<'a>(call: &'a ruff_python_ast::ExprCall, environment: &Environment) -> Option<&'a str> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let root = attribute_chain_root_name(attribute.value.as_ref())?;
    if environment.read(root).is_some() {
        return None;
    }
    if MODELED_MODULE_NAMES.contains(&root) {
        return None;
    }
    Some(root)
}

/// `receiver.attr(...)` — the known receiver shapes this file
/// dispatches: `math.<name>(...)` / `re.compile(...)` (only when the
/// module name is not shadowed by a local binding) and a method call
/// on an evaluated receiver (an exact string's method, a dict's `.get`
/// or a view method, or a set method).
pub(in super::super) fn evaluate_attribute_call(
    attribute: &ruff_python_ast::ExprAttribute,
    arguments: &[AbstractValue],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if let Expr::Name(module_name) = attribute.value.as_ref() {
        if module_name.id.as_str() == "math" && environment.read("math").is_none() {
            if let Some(value) = math_models::math_call_result(attribute.attr.as_str(), arguments, kernel) {
                return value;
            }
            // `math_call_result` declined — for the eight domain-limited
            // names (`log`/`log2`/`log10`/`log1p`/`asin`/`acos`/`atanh`/
            // `acosh`), a STRADDLING operand still determines a value
            // over its served half, alongside the fire `possible_raise`
            // (`domain_limited_family_possible_raise`) pushes at the
            // sink — the same "the finding and the value both stand"
            // split `split_divisor_transfer` keeps for a sometimes-zero
            // divisor. An entirely-raising or unreadable operand still
            // answers `unknown()` here, unchanged.
            if let Some(family) = math_models::DomainLimitedFamily::of_function(attribute.attr.as_str()) {
                if let [only] = arguments {
                    if let Some(value) = math_models::domain_raise_served_half_value(family, only, kernel) {
                        return value;
                    }
                }
            }
            return unknown();
        }
        // `random.random()` — the sound `[0.0, 1.0)` range
        // (`math_models::random_call_result`'s own doc, citing
        // library/random.rst). Only this one function of the module is
        // modeled; every other `random.*` call falls through to the
        // generic unmodeled-call path below.
        if module_name.id.as_str() == "random" && environment.read("random").is_none() {
            if let Some(value) = math_models::random_call_result(attribute.attr.as_str(), arguments) {
                return value;
            }
        }
        // `re.compile(pattern)` — library/re.html, `re.compile`: "Compile
        // a regular expression pattern... into a regular expression
        // object." This domain has no Pattern kind (no regex-engine
        // knowledge is tracked), so the result is answered opaque —
        // the honest "a compiled pattern" sort, never a specific value
        // (b-body-expressions.py's `literal_regex`).
        if module_name.id.as_str() == "re" && environment.read("re").is_none() {
            if attribute.attr.as_str() == "compile" {
                return opaque_value("a compiled pattern");
            }
            // `re.match(pattern, string)` — library/re.html: "Return a
            // corresponding match object" (or None on no match). This
            // file cannot decide WHICH of the two outcomes a real regex
            // engine would reach (no pattern-matching engine is
            // modeled), so it answers the match-object sort ONLY, never
            // the None-on-no-match alternative — an honest over-
            // approximation of "some value came back," matching the
            // fixture row's own sort-mismatch framing
            // (c-reads-and-values.py's `string_match`).
            if attribute.attr.as_str() == "match" {
                return opaque_value("a match object");
            }
            // `re.search(pattern, string)` — library/re.html, `function::
            // search(pattern, string, flags=0)`: "Scan through string
            // looking for the first location where the regular
            // expression pattern produces a match, and return a
            // corresponding Match. Return None if no position in the
            // string matches the pattern." Modeled ONLY when `pattern`
            // is a known exact string containing NO regex metacharacter
            // (`is_literal_regex_pattern`'s own doc — `re.escape`'s own
            // documented special-character set) and `string` (the
            // subject) is a known exact string: a metacharacter-free
            // pattern's own regex semantics reduce to a plain SUBSTRING
            // test, decidable without a regex engine. A found substring
            // answers the match-object sort (the same
            // over-approximation `re.match` above already gives — this
            // file cannot build a real Match object); an ABSENT
            // substring answers the exact `None` CPython's own
            // `search` returns on no match — `null_value()`, matching
            // `dict.get`'s own None-on-absent shape.
            if attribute.attr.as_str() == "search" {
                if let [pattern, subject] = arguments {
                    if let Some(value) = re_search_literal_value(pattern, subject) {
                        return value;
                    }
                }
                return unknown();
            }
            // `re.sub(pattern, repl, string)` — library/re.html,
            // `function:: sub(pattern, repl, string, count=0, flags=0)`:
            // "Return the string obtained by replacing the leftmost
            // non-overlapping occurrences of pattern in string by the
            // replacement repl... with no count, every match is
            // replaced" — AGENT-BRIEF.md's own confirmed fact, the twin
            // of JS's GLOBAL `.replace(/…/g)`. Modeled the same way
            // `search` is: `pattern` and `repl` must both be known exact
            // strings with `pattern` metacharacter-free (the same
            // `is_literal_regex_pattern` gate), so the whole call
            // reduces to `string.replace(pattern, repl)` —
            // `string_models::string_method_result`'s own `replace` row
            // already implements the every-occurrence semantics this
            // reduction needs.
            if attribute.attr.as_str() == "sub" {
                if let [pattern, repl, subject] = arguments {
                    if is_literal_regex_pattern(pattern) {
                        if let Some(value) = string_models::string_method_result("replace", subject, &[pattern.clone(), repl.clone()]) {
                            return value;
                        }
                    }
                }
                return unknown();
            }
            // `re.fullmatch(pattern, s)` / `re.finditer(pattern, s)` READ
            // AS A VALUE (as opposed to `narrowing.rs`'s own truthy-
            // condition-only reading of the identical calls) —
            // library/re.html: `fullmatch(pattern, string)`, "If the
            // whole string matches... return a corresponding match
            // object"; `finditer(pattern, string)`, "Return an iterator
            // yielding match objects." Modeled ONLY for a known
            // exact-string pattern argument (`string_models::
            // match_object_value`'s own doc — the same literal-pattern
            // gate `narrow_regex_module_call` already applies for its
            // own truthy-condition reading). Both calls answer the same
            // match value: every group a match carries — the whole
            // match included — is the text that match SPANS, so the two
            // callers differ in WHICH subjects produce a match, never in
            // what a produced match's own groups read back as. For
            // `finditer` the single MATCH this call answers stands in
            // for "some element of the iterator," the value a `for m in
            // re.finditer(...)` loop's OWN iterable-element read needs.
            // `loops.rs::iterable_values`'s `Expr::Call` arm has no
            // recognizer for `re.finditer` today, so a `for` loop over
            // this call still declines upstream of ever reaching this
            // dispatch — this row serves a NON-loop value use
            // (`m = re.fullmatch(...); m.group(1)`) end to end, and
            // stands ready for the loop element path once that other
            // file's own recognizer is added.
            if matches!(attribute.attr.as_str(), "fullmatch" | "finditer") {
                if let [pattern, _subject] = arguments {
                    if let Some(pattern_text) = exact_string_values(pattern).and_then(code_points_to_string) {
                        if let Some(value) = string_models::match_object_value(&pattern_text) {
                            return value;
                        }
                    }
                }
                return unknown();
            }
        }
        // `json.loads(s)` — library/json.rst, `function:: loads(s, ...)`:
        // "deserialize s... to a Python object using this conversion
        // table" (the JSON-to-Python table this function's own doc
        // cites). Modeled for a known exact-string `s` whose text is one
        // of the JSON SCALAR productions this file parses by hand
        // (`json_scalar_literal_value`'s own doc: an integer, a float, a
        // quoted string, `true`/`false`/`null`) — the corpus's own rows
        // never need array/object parsing, so that grammar is not built.
        // An `s` this file holds no fact about (an opaque string — the
        // ISSUES.md b-runners:124 row) answers `json_loads_value_space`
        // instead of bare `unknown()`: every shape `loads` can return is
        // ONE determined claim, never a narrower guess this file cannot
        // back (a Float-sorted answer would be false whenever the real
        // payload is a dict/list/str/bool/None).
        if module_name.id.as_str() == "json" && environment.read("json").is_none() {
            if attribute.attr.as_str() == "loads" {
                if let [text] = arguments {
                    if let Some(carried) = json_grammar::round_trip_carried_value(text) {
                        return carried.clone();
                    }
                    if let Some(text) = exact_string_values(text).and_then(code_points_to_string) {
                        if let Some(value) = json_scalar_literal_value(text.trim()) {
                            return value;
                        }
                    }
                }
                return json_loads_value_space();
            }
            // `json.dumps(obj)` — library/json.rst, `function::
            // dumps(obj, ...)`: "Serialize obj to a JSON formatted str
            // using this conversion table" (the Python-to-JSON table
            // this function's own doc cites), default `separators =
            // (', ', ': ')` (no `indent`). Modeled for a known exact
            // string, a known Integer, or a known `Kind::Object` whose
            // OWN values are each one of those same JSON-serializable
            // shapes (`json_dumps_value`'s own doc) — every other value
            // shape (Float, Boolean, a nested list, an unknown value)
            // declines the whole call.
            //
            // A `Kind::Object` this exact reading cannot fully resolve
            // (a member carries a SET rather than one concrete value —
            // a windowed int, a `Literal[...]` string union) falls to
            // `json_grammar::dumps_grammar` instead: the serialized
            // text's GRAMMAR, composed member by member from the same
            // conversion table, wrapped as a string-sorted `Kind::Set`
            // so a downstream length/pattern sink still judges it
            // (`json_grammar.rs`'s own doc states the full rule).
            if attribute.attr.as_str() == "dumps" {
                if let [value] = arguments {
                    if let Some(text) = json_dumps_value(value) {
                        return string_models::string_literal_value(&text);
                    }
                    if let Some(grammar) = json_grammar::dumps_grammar(value, &mut json_dumps_value) {
                        return grammar;
                    }
                    if let Some(carrier) = json_grammar::dumps_round_trip_carrier_value(value) {
                        return carrier;
                    }
                }
                return unknown();
            }
        }
        // `importlib.import_module(name)` — library/importlib.html,
        // `function:: import_module(name, package=None)`: "Import a
        // module... and return the imported module." This domain has
        // no module-object Kind (the same "no dedicated kind" posture
        // `type(object)`'s own opaque answer takes,
        // `builtin_models::builtin_call_result`'s `"type"` row) — the
        // honest answer is the opaque "a module object" sort, never a
        // specific value: d-module-surface.py's own `dynamic_import`
        // row states exactly that reason ("a module object is never an
        // Age"). Modeled regardless of the argument shape (a dynamic
        // import's own module identity is never read further by this
        // corpus's rows — only that the RESULT is opaque, not a refined
        // scalar).
        if module_name.id.as_str() == "importlib" && environment.read("importlib").is_none() {
            if attribute.attr.as_str() == "import_module" {
                return opaque_value("a module object");
            }
        }
        // `types.MappingProxyType(d)` — library/stdtypes.rst, "Mapping
        // Types — dict": types.MappingProxyType "wraps" a dict in a
        // read-only VIEW; a READ through the proxy returns exactly the
        // wrapped dict's own value (this row only reads — WRITING
        // through the proxy raises `TypeError`, a genuine CPython
        // divergence from a JS `Object.freeze` wrapper this file does
        // not model since no row exercises a write). Answered as the
        // IDENTITY of its one known `Kind::Object` argument — the same
        // pass-through `builtin_models::cast_call` already gives
        // `typing.cast`'s second argument, reused here rather than
        // building a second "read-only dict" tag no reader would ever
        // distinguish from a plain dict (every subscript/`.get()` read
        // this file models is already non-mutating).
        if module_name.id.as_str() == "types" && environment.read("types").is_none() {
            if attribute.attr.as_str() == "MappingProxyType" {
                if let [dict] = arguments {
                    if dict.kind == Kind::Object {
                        return dict.clone();
                    }
                }
                return unknown();
            }
        }
        // `weakref.WeakSet()` / `weakref.WeakKeyDictionary()` — the
        // BARE, zero-argument constructor form only (library/weakref.rst:
        // both classes hold "weak references to its elements/keys," a
        // fact invisible to any reader that only ever consumes the
        // collection via containment/subscript, matching
        // `collection_models.rs`'s own "a set's uniqueness is invisible
        // to a len()/iteration reader" note for an ordinary `set`).
        // `WeakSet()` answers the same empty-list `Kind::List` a bare
        // `set()` does (`builtin_models::set_constructor_call`'s own
        // zero-argument row); `WeakKeyDictionary()` answers the same
        // empty-dict `Kind::Object` a bare `dict()` does. Neither
        // constructor call takes a required argument this file reads —
        // a call WITH an argument (copying from an existing mapping)
        // falls through to `unknown()`, not modeled.
        if module_name.id.as_str() == "weakref" && environment.read("weakref").is_none() {
            if attribute.attr.as_str() == "WeakSet" && arguments.is_empty() {
                return collection_models::list_literal_value(&[]);
            }
            if attribute.attr.as_str() == "WeakKeyDictionary" && arguments.is_empty() {
                return collection_models::dict_literal_value(&[], &[]);
            }
        }
        // `await asyncio.gather(a, b, ...)` — library/asyncio-task.rst,
        // `awaitablefunction:: gather(*aws, ...)`: "If all awaitables
        // are completed successfully, the result is an aggregate list
        // of returned values. The order of result values corresponds
        // to the order of awaitables." Each positional argument here is
        // already the settled value the caller's own `await`/call
        // evaluation produced (a same-module coroutine call summarizes
        // through the ordinary call dispatch above, `async`/`await`
        // carrying no gate of their own — `evaluate_expression`'s
        // `Expr::Await` arm passes its inner value straight through),
        // so this row only needs to collect the already-evaluated
        // arguments into the aggregate List `asyncio.gather` documents.
        // `return_exceptions=`/other keyword arguments are not modeled
        // (the call-site keyword guard above this function's own
        // caller already declines a call carrying any keyword
        // argument).
        if module_name.id.as_str() == "asyncio" && environment.read("asyncio").is_none() {
            if attribute.attr.as_str() == "gather" {
                return collection_models::list_literal_value(arguments);
            }
        }
        // `time.time()` / `os.open(path, flags)` / `os.close(fd)` /
        // `unicodedata.normalize(form, s)` — `builtin_models::
        // stdlib_call_result`'s own doc states each row's citation.
        if module_name.id.as_str() == "time" && environment.read("time").is_none() {
            if let Some(value) = builtin_models::stdlib_call_result("time", attribute.attr.as_str(), arguments) {
                return value;
            }
        }
        if module_name.id.as_str() == "os" && environment.read("os").is_none() {
            if let Some(value) = builtin_models::stdlib_call_result("os", attribute.attr.as_str(), arguments) {
                return value;
            }
        }
        if module_name.id.as_str() == "unicodedata" && environment.read("unicodedata").is_none() {
            if let Some(value) = builtin_models::stdlib_call_result("unicodedata", attribute.attr.as_str(), arguments) {
                return value;
            }
        }
        if module_name.id.as_str() == "dict" && environment.read("dict").is_none() {
            if let Some(value) = builtin_models::stdlib_call_result("dict", attribute.attr.as_str(), arguments) {
                return value;
            }
        }
        // `base64.b64encode(s)` — `bytes_models::base64_call_result`'s
        // own doc (library/base64.rst).
        if module_name.id.as_str() == "base64" && environment.read("base64").is_none() {
            if let Some(value) = bytes_models::base64_call_result(attribute.attr.as_str(), arguments) {
                return value;
            }
        }
    }
    // `datetime.datetime.now()` — the receiver (`attribute.value`) is
    // either a TWO-level attribute chain (`Attribute(value=Attribute
    // (value=Name("datetime"), attr="datetime"), attr="now")`) when
    // `datetime` reached the file qualified, or a bare aliased class
    // name (`dt.now()`, `from datetime import datetime as dt`) ONE
    // level — never reaching `is_datetime_datetime_attribute`'s own
    // CONSTRUCTION-callee use (that check ALSO gates
    // `datetime.datetime(...)`, whose `call.func` IS the receiver
    // chain itself; here `attribute` is one level further out,
    // `datetime.datetime.now`/`dt.now`). classmethod:: datetime.now
    // (tz=None): "Return the current local date and time." — a value
    // that changes every run, never a whole number Age could ever
    // admit (this fixture's own row's reason: "the current moment is
    // not in the set"); answered OPAQUE, the same "not a scalar/set
    // this domain models" honesty every other host-nondeterministic
    // read in this file already carries. The `tz=` argument (if any)
    // is not read — every outcome is equally opaque regardless of
    // which timezone the caller requests.
    if is_datetime_datetime_attribute(attribute.value.as_ref(), environment) && attribute.attr.as_str() == "now" {
        return opaque_value("the current datetime");
    }
    // `datetime.datetime.strptime(date_string, format)` — date.12,
    // the SAME receiver shape `.now()` reads just above (qualified
    // chain or bare aliased class name). Modeled ONLY when BOTH
    // arguments are known exact strings; a NON-literal `format` (a
    // parameter, a computed expression, an f-string) is not a string
    // this file can read the DIRECTIVES of at all, so it declines the
    // same way `date_fromisoformat_value`'s own non-literal-argument
    // row does — no sentence-carrying channel exists on this dispatch
    // path (`evaluate_attribute_call` returns a plain `AbstractValue`,
    // never a message), so every decline reason below is named in
    // this comment, the same way every other declining recognizer in
    // this file states its reason in prose beside its own `return
    // unknown()`.
    //
    // The exact literal `"%Y-%m-%d"` keeps STAGE 1's own path
    // (`strptime_iso_date_value`, reusing `date_fromisoformat_value`
    // outright rather than re-deriving its parse). Every OTHER
    // literal format is read by STAGE 2's directive scanner
    // (`strptime2_scan_format`): a format whose every directive is
    // one of `%Y %m %d %H %M %S %f %j %U %W %y %z %%` is parsed by
    // `strptime2_parse` against the SAME two kernel asks STAGE 1
    // poses (`pyYearInRange` then `validDate`); a format naming a
    // directive this round does not transcribe outright declines,
    // naming that ONE directive rather than the whole format string —
    // split into the three decline shapes `Strptime2Decline` states: a
    // locale directive (`%A %b %B %p %c %x %X`, datetime.rst note (1)
    // — no host-independent value set exists for these AT ALL); an
    // unread directive (`%Z %I %G %u %V` — not yet transcribed against
    // the spec, but buildable); and `%a` alone, its own
    // `WeekdayAbbreviation` case — its C-locale value set IS host-
    // independent (`read_weekday_abbreviation_field`'s own doc), so
    // `%a` is scanned as an ACCEPTED directive exactly when this
    // module's own `locale_never_set` premise holds
    // (`environment.locale_never_set()`, `module_never_calls_setlocale`'s
    // own doc — POSIX runs the portable `'C'` locale, whose weekday
    // abbreviations ARE this fixed ASCII set, unless the module ever
    // calls `locale.setlocale`); `None`/`Some(false)` both keep `%a`
    // as a first-blocking directive the same as before this premise
    // existed, never assuming the C locale without the fact in hand.
    let accept_weekday_abbreviation = environment.locale_never_set().unwrap_or(false);
    if is_datetime_datetime_attribute(attribute.value.as_ref(), environment) && attribute.attr.as_str() == "strptime" {
        if let [text, format] = arguments {
            if let (Some(text_points), Some(format_points)) = (exact_string_values(text), exact_string_values(format)) {
                if let (Some(text_spelling), Some(format_spelling)) = (code_points_to_string(text_points), code_points_to_string(format_points)) {
                    if format_spelling == "%Y-%m-%d" {
                        return match strptime_iso_date_value(&text_spelling, kernel) {
                            Some(value) => value,
                            None => unknown(),
                        };
                    }
                    if strptime2_scan_format(&format_spelling, accept_weekday_abbreviation).is_ok() {
                        return match strptime2_parse(&format_spelling, &text_spelling, kernel) {
                            Some(value) => value,
                            None => unknown(),
                        };
                    }
                    // strptime2_scan_format's own Err names the
                    // specific unread or locale directive that
                    // blocked this format — no channel carries that
                    // reason string further than this comment today
                    // (the same standing decline every other row in
                    // this dispatch takes), so the value answer is
                    // unknown() regardless of which of the two
                    // Strptime2Decline arms fired.
                }
            }
        }
        return unknown();
    }
    let receiver = evaluate_expression(&attribute.value, environment, kernel);
    // A tagged `datetime_datetime` instance's own METHODS —
    // `.timestamp()` (exact, aware-UTC-only, `datetime_timestamp_value`'s
    // own doc) and `.isoformat()` (exact — the kernel's `isoDateText`
    // render for the date half plus the instance's own already-known
    // clock fields and offset, `datetime_isoformat_value`'s own doc).
    if receiver.kind == Kind::Object && receiver.source == "datetime_datetime" {
        if attribute.attr.as_str() == "timestamp" && arguments.is_empty() {
            return match datetime_timestamp_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        if attribute.attr.as_str() == "isoformat" {
            if arguments.is_empty() {
                if let Some(value) = datetime_isoformat_value(&receiver, kernel) {
                    return value;
                }
            }
            // A `sep=`/`timespec=` argument changes the spelling this
            // render does not thread through, and an instance whose
            // offset this crate never resolved carries no exact text —
            // both keep the sort-only claim datetime.rst's own format
            // still proves.
            return opaque_value("an ISO 8601 datetime string");
        }
    }
    // A tagged `datetime_date` instance's own METHODS — `.weekday()`
    // (date.8, Monday 0), `.isoweekday()` (date.8, Monday 1),
    // `.toordinal()` (date.9), `.isocalendar()` (date.10) — each exact,
    // each posing its own dedicated kernel ask directly (see
    // `date_weekday_value`/`date_toordinal_value`/
    // `date_isocalendar_value`'s own docs for the exact op).
    if receiver.kind == Kind::Object && receiver.source == "datetime_date" {
        if attribute.attr.as_str() == "weekday" && arguments.is_empty() {
            return match date_weekday_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        if attribute.attr.as_str() == "isoweekday" && arguments.is_empty() {
            return match date_isoweekday_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        if attribute.attr.as_str() == "toordinal" && arguments.is_empty() {
            return match date_toordinal_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        if attribute.attr.as_str() == "isocalendar" && arguments.is_empty() {
            return match date_isocalendar_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        // `.isoformat()` — date.12's render direction, the kernel's own
        // `isoDateText` arm through `date_isoformat_value`.
        if attribute.attr.as_str() == "isoformat" && arguments.is_empty() {
            return match date_isoformat_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        // `.strftime(format)` — date.12 STAGE 1. Recognized (the method
        // NAME matches, `format` is a known exact string) so a
        // non-`"%Y-%m-%d"` literal or a computed format can each name
        // their own reason below rather than fall through unrecognized.
        if attribute.attr.as_str() == "strftime" {
            if let [format] = arguments {
                if let Some(format_points) = exact_string_values(format) {
                    if let Some(format_spelling) = code_points_to_string(format_points) {
                        if format_spelling == "%Y-%m-%d" {
                            return match strftime_iso_date_value(&receiver, kernel) {
                                Some(value) => value,
                                None => unknown(),
                            };
                        }
                        // a literal format that is not `"%Y-%m-%d"` —
                        // date.12 STAGE 2's own directive-grammar
                        // kernel theory, not built by this stage
                    }
                }
                // a non-literal (computed) format — this file cannot
                // read the directive sequence of an expression it
                // cannot fold to an exact string at all
            }
            return unknown();
        }
    }
    let receiver_is_exact_string = exact_string_values(&receiver).is_some();
    if receiver_is_exact_string {
        if let Some(value) = string_models::string_method_result(attribute.attr.as_str(), &receiver, arguments) {
            return value;
        }
        // The exact row declined — the receiver's own content is known,
        // but some ARGUMENT is not (`"-".join(parts)` for an unread
        // `parts`, `s.replace(old, new)` for an unread `old`). The
        // method's own contract still names the SORT of what it
        // returns, so the sort-only rows below get their turn before
        // this call falls to `unknown()`; an exact receiver is a
        // string-sorted receiver, so it qualifies for exactly the same
        // claims an unread string-sorted one does.
        if let Some(value) = string_models::string_method_sort_only_result(attribute.attr.as_str(), &receiver, arguments) {
            return value;
        }
        if let Some(value) = string_models::string_method_int_sort_only_result(attribute.attr.as_str(), arguments) {
            return value;
        }
        return unknown();
    }
    // A STRING-SORTED but NOT EXACT receiver (`s: str`, unrefined — a
    // bare-`str` parameter seeds the whole-strings ground,
    // `typereading::base_sort_return_refinement`'s own doc — or any
    // other Set-shaped string value this file's readers already
    // produced): `exact_string_values` above already declined, but a
    // method whose own CPython contract always returns ANOTHER `str`
    // (or, for `find`/`index`, an `int`) still states that SORT exactly,
    // the same "answer the sort, not a guessed value" row `math_models`'s
    // approximated family keeps for a numeric transcendental over a
    // known window. `assignability::states_sequence`/`sequence_shaped`
    // is the SAME string-vs-numeric-ground test that file's own sort
    // laws already gate on — never a second recognizer for the same
    // question.
    if receiver.kind == Kind::Set
        && (assignability::states_sequence(&receiver.set) || assignability::sequence_shaped(&receiver.set))
    {
        if let Some(value) = string_models::string_method_sort_only_result(attribute.attr.as_str(), &receiver, arguments) {
            return value;
        }
        if let Some(value) = string_models::string_method_int_sort_only_result(attribute.attr.as_str(), arguments) {
            return value;
        }
    }
    // `dict[str, X]` PARAMETER'S own unbounded-key receiver
    // (`Kind::ObjectStar` — `check.rs::seed_parameters`'
    // `dict_star_value_seed`/`known_dict_star`): `.get(key)` is the ONE
    // method this shape answers (`collection_models::dict_get_result`'s
    // own dict-star arm) — checked ahead of, and separately from, the
    // `Kind::Object` block below, since `.setdefault`/the dict-view
    // methods that block also handles all assume a CLOSED receiver
    // (`mutated_receiver`/`dict_with_item`/`dict_view_method_result`
    // all read `container.keys` directly) and must never see an
    // unbounded-key receiver.
    if receiver.kind == Kind::ObjectStar && attribute.attr.as_str() == "get" {
        let key = arguments.first();
        let default = arguments.get(1);
        if let Some(key) = key {
            return match collection_models::dict_get_result(&receiver, key, default) {
                Some(value) => value,
                None => unknown(),
            };
        }
        return unknown();
    }
    if receiver.kind == Kind::Object {
        // `match.group(n)` / `match.group("name")` —
        // library/re.html#re.Match.group, the ONE-ARGUMENT form, by
        // number or by symbolic group name — on a `string_models::
        // match_object_value`-built receiver (`MATCH_WITH_GROUPS_WORD`-
        // tagged, the `re.fullmatch`/`re.finditer` value-producing arm
        // above). `string_models::matched_group_grammar`'s own doc
        // states the exact group reading; any other receiver
        // (the bare `opaque_value("a match object")` `re.match`/
        // `re.search` still answer) or argument shape declines to
        // `unknown()`, unchanged from today.
        if attribute.attr.as_str() == "group" {
            if let Some(value) = string_models::matched_group_grammar(&receiver, arguments) {
                return value;
            }
        }
        // `<bytes-like>.decode()` — library/stdtypes.html#bytes.decode,
        // the zero-argument default-encoding form — on a
        // `bytes_models::base64_call_result`-built receiver
        // (`BASE64_ENCODED_WORD`-tagged). `bytes_models::
        // bytes_decode_call`'s own doc states the exact reading; any
        // other bytes-like receiver (a plain `Kind::List` bytes value,
        // or an `.encode()` result's own `ENCODED_BYTES_WORD` tag)
        // declines here, matching that function's own scope.
        if attribute.attr.as_str() == "decode" {
            if let Some(value) = bytes_models::bytes_decode_call(&receiver, arguments) {
                return value;
            }
        }
        if attribute.attr.as_str() == "get" {
            // dict.get(key, default=None, /) — a missing default argument
            // is None (stdtypes.rst, dict's `method:: get`), matching
            // `dict_get_result`'s own `None` reading of an absent default
            let key = arguments.first();
            let default = arguments.get(1);
            if let Some(key) = key {
                return match collection_models::dict_get_result(&receiver, key, default) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            return unknown();
        }
        // dict.setdefault(key, default=None) READ as a VALUE (not
        // through the statement-level mutation sink,
        // `collection_models::mutated_receiver`'s own `setdefault`
        // arm): "If key is in the dictionary, return its value. If not,
        // insert key with a value of default and return default"
        // (stdtypes.rst, dict's method:: setdefault) — the VALUE half
        // of that contract is identical to `dict.get(key, default)`'s
        // own present-wins-over-default row, so this arm reuses
        // `dict_get_result` directly rather than re-derive the same
        // present/absent branch a second time. The receiver's own
        // write (extending it on a miss) is `mutated_receiver`'s
        // business when this call sits in a statement-level write
        // position; a nested read like this one only ever needs the
        // answered value.
        if attribute.attr.as_str() == "setdefault" {
            let key = arguments.first();
            let default = arguments.get(1);
            if let Some(key) = key {
                return match collection_models::dict_get_result(&receiver, key, default) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            return unknown();
        }
        if arguments.is_empty() {
            if let Some(value) = dict_view_method_result(attribute.attr.as_str(), &receiver) {
                return value;
            }
        }
    }
    // `(<a known Float/Integer>).is_integer()` — stdtypes.rst, `method::
    // float.is_integer()`: "Return True if the float instance is finite
    // with integral value, and False otherwise" (`int.is_integer()`,
    // added 3.12, "Returns True" always — "Exists for duck type
    // compatibility"). Exact for any known single numeric receiver: an
    // Integer-sorted receiver is always `True` (the int row); a
    // Float-sorted receiver checks `fract() == 0.0 && is_finite()`
    // directly on the known f64.
    if attribute.attr.as_str() == "is_integer" && arguments.is_empty() {
        if let Some((value, sort)) = single_numeric_value(&receiver) {
            let is_integer = sort == PrimitiveKind::Integer || (value.is_finite() && value.fract() == 0.0);
            return boolean_answer(is_integer);
        }
    }
    if receiver.kind == Kind::List {
        if let Some(value) = set_method_result(attribute.attr.as_str(), &receiver, arguments) {
            return value;
        }
        // `<known bytes>.decode("ascii")` — a `bytes` literal is the
        // same `Kind::List` of known byte ints an ordinary list literal
        // builds (`bytes_models.rs`'s own module doc), so its `.decode`
        // is answered here rather than in the `Kind::Object` arm above,
        // which serves only the `base64.b64encode`-tagged receiver.
        // `bytes_models::bytes_decode_call`'s own doc states which byte
        // sequences and encodings it decides.
        if attribute.attr.as_str() == "decode" {
            if let Some(value) = bytes_models::bytes_decode_call(&receiver, arguments) {
                return value;
            }
        }
        // `list.pop()`/`list.pop(i)` READ as a VALUE (not through the
        // statement-level mutation sink, `collection_models::
        // mutated_receiver`'s own `pop` arm): "retrieves the item at *i*
        // and also removes it from *s*" (stdtypes.rst's Mutable-Sequence-
        // Types table, `s.pop([i])`) — c-reads-and-values.py's
        // `list_pop`'s own RHS shape, `overs.pop()` used directly as a
        // `return` expression rather than first bound to a name. Only the
        // RESULT half of `mutated_receiver`'s `(new receiver, result)`
        // pair is read here: the receiver's own shrink is the write
        // sink's business (`walk_mutating_call_statement`'s statement-
        // level rebind), the same "fires/writes belong to the sink" split
        // every other nested value read in this file already draws
        // (the construction and instance-method-call arms above,
        // `dict.setdefault`'s own value-read arm just above this one).
        if attribute.attr.as_str() == "pop" {
            if let Some((_new_receiver, result)) = collection_models::mutated_receiver("pop", &receiver, arguments) {
                return result;
            }
            return unknown();
        }
        // `xs.sort()` READ AS A VALUE (not through the statement-level
        // mutation sink): "This method sorts the list in place... This
        // method modifies the sequence in place for economy of space
        // when sorting a large sequence. To remind users that it
        // operates by side effect, it does not return the sorted
        // sequence" (stdtypes.rst's Mutable-Sequence-Types table,
        // `s.sort(...)`) — `None` ALWAYS, regardless of whether the
        // receiver's own elements are known (the trap this row exists
        // to name: reading the RETURN VALUE is always a sort mismatch
        // against a refined Age, never the sorted list itself). The
        // sorted LIST is `mutated_receiver`'s own business when this
        // call sits in a statement-level write position — this arm only
        // ever answers the call's own result, matching the "fires/writes
        // belong to the sink" split every other nested value read in
        // this file already draws.
        if attribute.attr.as_str() == "sort" && arguments.is_empty() {
            return null_value();
        }
        // `xs.index(needle)` — stdtypes.rst's Common Sequence Operations
        // table, `s.index(x)`: "index of the first occurrence of x in
        // s." Modeled only on the FOUND leg (a missing needle raises
        // ValueError at runtime instead of returning — that leg is
        // `call_provable_raise`'s own `"index"` row, checked separately
        // against the same `single_pair_equal` equality this row uses,
        // so the two passes agree on exactly which needle is present).
        // Answers the position of the first matching element as an
        // exact Integer.
        if attribute.attr.as_str() == "index" {
            if let [needle] = arguments {
                if let Some(position) = receiver.items.iter().position(|element| single_pair_equal(element, needle) == Some(true)) {
                    return known_values(vec![position as f64], PrimitiveKind::Integer, TrustProved);
                }
            }
        }
    }
    unknown()
}
