/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! RUNG 2 of the compiled-extension recognition ladder
//! (`packages/cpp/findings/python-c-extension-boundary.md`, the manifest
//! reader template — build order item 2): a per-module binding manifest,
//! a committed JSON table mapping FUNCTION NAME → ENTRY-CONTRACT STRING →
//! PRODUCER SYMBOL, read beside the checked file. A module named in a
//! manifest makes calls to its listed functions judge arguments against
//! the parsed entry contract — an argument escaping its declared sort
//! fires the same crossing-fit refusal shape the stdio edge fires
//! (`foreign_edge.rs`'s own `foreign_scalar_subset`, the pattern this
//! file's `entry_crossing_fits` mirrors rather than reuses, since the
//! stdio edge judges a `ForeignTsEntry`'s wire-decoded cases, not a
//! plain Sort word parsed from a signature string). The RETURN binds
//! NOTHING yet — the producer half is a later unit
//! (python-c-extension-boundary.md's build order item 3) — so a fitting
//! call still declines, naming the missing producer fact rather than
//! answering a value.
//!
//! A module named in NO manifest stays rung 1's plain named decline
//! (`expressions::unmodeled_module_call_name`,
//! `diagnostic_sentences::unmodeled_module_call`) — this file only
//! narrows that naming for a module that DOES have one.
//!
//! DISCOVERY: a manifest for module `widgets` is read from
//! `<entry_directory>/widgets.manifest.json` — beside the checked file,
//! the same directory `Environment::entry_directory` already carries
//! (`env.rs`'s own doc: mirrors `WalkContext::entry_directory`, the
//! checked file's own directory, the foreign-edge artifact cache's own
//! "beside the file that wrote it" convention, without that cache's
//! content-hash keying — a manifest is hand-authored and committed, not
//! machine-exported per checked file, so a plain sibling file is the
//! right shape). No `entry_directory` (a resolver-less test entry point)
//! discovers no manifest at all — a module still names rung 1's plain
//! decline in that case, never a crash.
//!
//! ENTRY-CONTRACT GRAMMAR: the PythonArgParser signature-string subset
//! `python-c-extension-boundary.md`'s own citation (PyTorch's
//! `THPVariable_<op>` argument strings) states as the template's
//! grammar:
//!
//! ```text
//! name(Sort arg1, Sort arg2=default, *, Sort kwarg=default)
//! ```
//!
//! `Sort` is one of `Scalar`, `int`, `float`, `bool`, `str` — parsed into
//! the numeric/string-ground `RefinedSet`s `typereading`'s own
//! constructors already build (`refined_sets::refinement_forms::numbers`/
//! `integer`, `refined_sets::codepoint_sets::strings`, `one_of` for
//! `bool`'s own `{0, 1}` — never a new set-shape constructor). A
//! malformed signature string declines the WHOLE manifest row (never a
//! partial parse): `parse_entry_contract` answers `None` for anything
//! the grammar does not exactly state.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::numbers;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::RefinedSet;
use serde_json::Value;

use crate::diagnostic_sentences;

/// The PythonArgParser subset's five admitted sorts — `Scalar` is the
/// unrestricted numeric ground (either int or float admitted, PyTorch's
/// own `Scalar` C++ type, which boxes either), `int`/`float` narrow it to
/// one Python numeric sort, `bool` is the two-element `{0, 1}` ground,
/// and `str` is the full string ground. No other spelling parses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestSort {
    Scalar,
    Int,
    Float,
    Bool,
    Str,
}

impl ManifestSort {
    /// The exact PythonArgParser spelling this sort parses from, and the
    /// word an entry-crossing fire names it by — the same spelling both
    /// ways, since the grammar's own words are already the plain English
    /// this checker's other sentences use.
    fn spelling(self) -> &'static str {
        match self {
            ManifestSort::Scalar => "Scalar",
            ManifestSort::Int => "int",
            ManifestSort::Float => "float",
            ManifestSort::Bool => "bool",
            ManifestSort::Str => "str",
        }
    }

    fn parse(word: &str) -> Option<ManifestSort> {
        match word {
            "Scalar" => Some(ManifestSort::Scalar),
            "int" => Some(ManifestSort::Int),
            "float" => Some(ManifestSort::Float),
            "bool" => Some(ManifestSort::Bool),
            "str" => Some(ManifestSort::Str),
            _ => None,
        }
    }

    /// This sort's own admitted `RefinedSet` — reusing `typereading`'s
    /// established constructors (`numbers()`/`integer()`/`strings()`),
    /// never a new set-shape. `bool`'s own ground is the two-element
    /// `one_of([0.0, 1.0])` — the same values `known_values(vec![0.0,
    /// 1.0], PrimitiveKind::Boolean, ...)` already carries for a Python
    /// `True`/`False` (`expressions.rs`'s own `BooleanLiteral` reading),
    /// so a Boolean-tagged argument's own two values sit exactly inside
    /// this ground.
    fn admitted_set(self) -> RefinedSet {
        match self {
            ManifestSort::Scalar => numbers(),
            ManifestSort::Int => make_refined_set(vec![integer()]),
            ManifestSort::Float => numbers(),
            ManifestSort::Bool => make_refined_set(vec![one_of(&[0.0, 1.0])]),
            ManifestSort::Str => strings(),
        }
    }

    /// Whether this sort is string-ground (`Str`) or numeric-ground
    /// (every other sort) — which kernel ask (`seq_subset` for a string
    /// pair, `scalar_subset` for a numeric pair) the crossing-fit check
    /// must send, mirroring `foreign_edge.rs::foreign_scalar_subset`'s
    /// own sequence-vs-scalar routing.
    fn is_string_ground(self) -> bool {
        matches!(self, ManifestSort::Str)
    }
}

/// One parameter position the entry contract states: its name, its sort,
/// whether it is keyword-only (after the `*` marker), and whether it
/// carries a default (making it optional at the call site — the template
/// reads the default's PRESENCE, never its own value, since the checker
/// judges only arguments a caller actually writes).
#[derive(Debug, Clone)]
pub struct ManifestParameter {
    pub name: String,
    pub sort: ManifestSort,
    pub keyword_only: bool,
    pub has_default: bool,
}

/// One function's whole parsed entry contract: its parameter list, in
/// the signature's own written order, and the PRODUCER SYMBOL the
/// manifest names for it — the C++/native symbol a later unit's producer
/// half will export a return fact for (build order item 3). The RETURN
/// itself is not stated here: the template's own scope is the ENTRY
/// only, per `python-c-extension-boundary.md`'s "the RETURN binds
/// nothing yet."
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    pub function_name: String,
    pub parameters: Vec<ManifestParameter>,
    pub producer_symbol: String,
}

/// One module's whole manifest: every function it lists, keyed by name.
#[derive(Debug, Clone)]
pub struct BindingManifest {
    pub module_name: String,
    pub entries: HashMap<String, ManifestEntry>,
}

/// Parses one PythonArgParser subset signature string into its parameter
/// list. Grammar: `name(Sort arg1, Sort arg2=default, *, Sort
/// kwarg=default)` — the leading `name(` and trailing `)` are read and
/// discarded (the function's OWN name is already the manifest's JSON key,
/// never re-read from here); a bare `*` marks every parameter AFTER it
/// keyword-only; `=default` marks a parameter optional, its default text
/// unread (this template judges only written arguments). ANY other shape
/// — an unrecognized Sort word, a parameter with no Sort at all, an
/// unbalanced paren, an empty parameter between two commas — declines
/// the WHOLE signature (`None`), never a partial parameter list: a
/// malformed entry contract is not a contract this checker can partially
/// trust.
pub fn parse_entry_contract(signature: &str) -> Option<Vec<ManifestParameter>> {
    let signature = signature.trim();
    let open = signature.find('(')?;
    if !signature.ends_with(')') {
        return None;
    }
    let inside = &signature[open + 1..signature.len() - 1];
    let inside = inside.trim();
    if inside.is_empty() {
        return Some(Vec::new());
    }
    let mut parameters = Vec::new();
    let mut keyword_only = false;
    for raw_parameter in inside.split(',') {
        let raw_parameter = raw_parameter.trim();
        if raw_parameter == "*" {
            keyword_only = true;
            continue;
        }
        if raw_parameter.is_empty() {
            return None;
        }
        let (typed_name, has_default) = match raw_parameter.split_once('=') {
            Some((typed_name, _default_text)) => (typed_name.trim(), true),
            None => (raw_parameter, false),
        };
        let mut words = typed_name.split_whitespace();
        let sort_word = words.next()?;
        let name = words.next()?;
        if words.next().is_some() {
            // a third word (`Sort name extra`) is not this grammar's
            // shape at all
            return None;
        }
        let sort = ManifestSort::parse(sort_word)?;
        parameters.push(ManifestParameter {
            name: name.to_owned(),
            sort,
            keyword_only,
            has_default,
        });
    }
    Some(parameters)
}

/// Reads `<entry_directory>/<module_name>.manifest.json` and parses every
/// function row it states — `Ok(manifest)` on a fully-readable file,
/// `Err(sentence)` naming what stopped it (the file does not exist, is
/// not readable JSON, or names an entry-contract string
/// `parse_entry_contract` declines). `Ok(None)` is never answered by this
/// function: a module with no manifest FILE at all is the caller's own
/// concern (`discover_manifest`), not a parse failure this function
/// reports.
fn read_manifest(manifest_path: &Path, module_name: &str) -> Result<BindingManifest, String> {
    let manifest_path_words = manifest_path.display().to_string();
    let raw = std::fs::read(manifest_path)
        .map_err(|err| diagnostic_sentences::manifest_unreadable(&manifest_path_words, &err.to_string()))?;
    let parsed: Value = serde_json::from_slice(&raw)
        .map_err(|err| diagnostic_sentences::manifest_unreadable(&manifest_path_words, &err.to_string()))?;
    let Some(functions) = parsed.as_object() else {
        return Err(diagnostic_sentences::manifest_unreadable(
            &manifest_path_words,
            "the manifest's own top level is not a JSON object mapping function name to its row",
        ));
    };
    let mut entries = HashMap::with_capacity(functions.len());
    for (function_name, row) in functions {
        let Some(row) = row.as_object() else {
            return Err(diagnostic_sentences::manifest_unreadable(
                &manifest_path_words,
                &format!("the row for '{function_name}' is not a JSON object"),
            ));
        };
        let Some(entry_contract) = row.get("entry").and_then(Value::as_str) else {
            return Err(diagnostic_sentences::manifest_unreadable(
                &manifest_path_words,
                &format!("the row for '{function_name}' states no \"entry\" signature string"),
            ));
        };
        let Some(producer_symbol) = row.get("producer").and_then(Value::as_str) else {
            return Err(diagnostic_sentences::manifest_unreadable(
                &manifest_path_words,
                &format!("the row for '{function_name}' states no \"producer\" symbol"),
            ));
        };
        let Some(parameters) = parse_entry_contract(entry_contract) else {
            return Err(diagnostic_sentences::manifest_unreadable(
                &manifest_path_words,
                &format!(
                    "the row for '{function_name}' states an entry contract this reader's grammar does not \
                    parse: '{entry_contract}'"
                ),
            ));
        };
        entries.insert(
            function_name.clone(),
            ManifestEntry {
                function_name: function_name.clone(),
                parameters,
                producer_symbol: producer_symbol.to_owned(),
            },
        );
    }
    Ok(BindingManifest { module_name: module_name.to_owned(), entries })
}

/// Memoizes `read_manifest` by its own file path — the walk reaches one
/// call site many times across a module's own many statements/branches,
/// and each reach would otherwise re-read and re-parse the same file.
/// Mirrors `foreign_edge_artifact.rs::foreign_artifacts`'s own memo
/// shape, minus the mtime-freshness re-check that reader's own doc
/// explains: a manifest is a COMMITTED, hand-authored file, never
/// machine-regenerated mid-session the way a foreign-edge artifact's own
/// producer output is, so this memo holds for the process's whole
/// lifetime.
fn manifest_cache() -> &'static std::sync::Mutex<HashMap<PathBuf, Result<Arc<BindingManifest>, String>>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<PathBuf, Result<Arc<BindingManifest>, String>>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Discovers and reads `module_name`'s own manifest beside
/// `entry_directory`, memoized. `None` when `entry_directory` itself is
/// absent (a resolver-less test entry point — nothing to discover
/// against) OR the manifest file does not exist at all (a module with NO
/// manifest is not an error; it is rung 1's own plain decline). `Some(Err
/// (sentence))` for a manifest FILE that exists but could not be fully
/// read/parsed; `Some(Ok(manifest))` for a fully-readable one.
pub fn discover_manifest(module_name: &str, entry_directory: Option<&Path>) -> Option<Result<Arc<BindingManifest>, String>> {
    let entry_directory = entry_directory?;
    let manifest_path = entry_directory.join(format!("{module_name}.manifest.json"));
    if !manifest_path.exists() {
        return None;
    }
    let mut cache = manifest_cache().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(cached) = cache.get(&manifest_path) {
        return Some(cached.clone());
    }
    let read = read_manifest(&manifest_path, module_name).map(Arc::new);
    cache.insert(manifest_path, read.clone());
    Some(read)
}

/// A crossing argument's own readable words for a fire message — this
/// lane's own compact vocabulary (never `assignability.rs`'s full sort
/// derivation, since a manifest entry states a plain Sort word, not a
/// full `DeclaredRefinement`): the PrimitiveKind tag when `value` is an
/// exact word, the opaque `kind_word` when the value carries one, and a
/// plain structural word (`"a dict"`, `"a list"`, `"None"`) otherwise.
fn crossing_value_words(value: &AbstractValue) -> String {
    match value.kind {
        Kind::Values => match value.kind_tag {
            Some(PrimitiveKind::Integer) => "an int".to_owned(),
            Some(PrimitiveKind::Float) => "a float".to_owned(),
            Some(PrimitiveKind::Boolean) => "a bool".to_owned(),
            Some(PrimitiveKind::String) => "a str".to_owned(),
            _ => "a number".to_owned(),
        },
        Kind::Set => match value.kind_tag {
            Some(PrimitiveKind::String) => "a str".to_owned(),
            Some(PrimitiveKind::Float) => "a float".to_owned(),
            _ => "a number".to_owned(),
        },
        Kind::Object => value.kind_word.map(str::to_owned).unwrap_or_else(|| "a dict".to_owned()),
        Kind::List => "a list".to_owned(),
        Kind::Null => "None".to_owned(),
        _ => "a value this checker cannot yet read".to_owned(),
    }
}

/// Whether `value` fits `sort`'s own admitted ground — the crossing-fit
/// check. `Ok(true)` fits, `Ok(false)` is a real refutation the caller
/// fires, `Err(())` is a kernel refusal this lane leaves UNJUDGED (never
/// refuted) — the same "a kernel that cannot decide leaves the crossing
/// unjudged rather than refuting it" posture
/// `foreign_edge.rs::foreign_scalar_subset`'s own doc states. Only
/// `Kind::Values`/`Kind::Set` values carry an exact word/`RefinedSet` this
/// check can ask the kernel about; every other value `Kind` (Object,
/// List, Null, Unknown) is judged by STRUCTURAL mismatch directly,
/// without a kernel ask at all — a dict/list/None is never a number or a
/// string regardless of what either ground states. A `Kind::Values`
/// operand asks `kernel.member` directly over its own exact `values`
/// slice — the same ask `assignability.rs::judge`'s own Values arm
/// already makes (one whole-word ask for a String-tagged value, one
/// per-value ask otherwise) — rather than building a singleton
/// `RefinedSet` to feed `scalar_subset`/`seq_subset`, since `member` is
/// the more precise decider for an EXACT word and is already proved for
/// both scalar and sequence-shaped grounds.
fn entry_crossing_fits(value: &AbstractValue, sort: ManifestSort, kernel: &Arc<RefinedTSKernel>) -> Result<bool, ()> {
    let admitted = sort.admitted_set();
    match value.kind {
        Kind::Object | Kind::List | Kind::Null => Ok(false),
        Kind::Unknown => Err(()),
        Kind::Values => {
            let is_string_value = value.kind_tag == Some(PrimitiveKind::String);
            if is_string_value != sort.is_string_ground() {
                return Ok(false);
            }
            if is_string_value {
                crate::kernel_ask::ask_kernel(|| (kernel.member)(&admitted, &value.values)).map_err(|_| ())
            } else {
                for v in &value.values {
                    match crate::kernel_ask::ask_kernel(|| (kernel.member)(&admitted, std::slice::from_ref(v))) {
                        Ok(true) => {}
                        Ok(false) => return Ok(false),
                        Err(_) => return Err(()),
                    }
                }
                Ok(true)
            }
        }
        Kind::Set => {
            let is_string_value =
                value.kind_tag == Some(PrimitiveKind::String) || refined_sets::codepoint_sets::is_string_ground(&value.set);
            if is_string_value != sort.is_string_ground() {
                return Ok(false);
            }
            if is_string_value {
                crate::kernel_ask::ask_kernel(|| (kernel.seq_subset)(&value.set, &admitted)).map_err(|_| ())
            } else {
                crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&value.set, &admitted)).map_err(|_| ())
            }
        }
        _ => Err(()),
    }
}

/// One judged call's own outcome: every entry-crossing FIRE (an RTS7001
/// message and the argument's own range), and the sentence the RETURN
/// declines with — "the manifest names its entry but no producer exports
/// its return fact," naming both the call and the producer symbol
/// (`diagnostic_sentences::manifest_entry_names_no_producer`), since the
/// producer half is a later unit.
pub struct ManifestCallOutcome {
    pub fires: Vec<(ruff_text_size::TextRange, String)>,
    pub return_decline_sentence: String,
}

/// Judges a recognized manifest call's own positional and keyword
/// arguments against `entry`'s own parsed contract, answering the fires
/// and the return's own decline sentence. `positional`/`keyword` are the
/// call's own ALREADY-EVALUATED arguments, paired with the AST range each
/// one came from (`sink_value`'s own call site builds these the same way
/// `positional_arguments_for_def` already pairs an evaluated value with
/// its own source range for a Fire to anchor to). A positional argument
/// past the contract's own positional-parameter count, or a keyword
/// naming no parameter the contract states, is NOT judged here — an
/// arity mismatch is a different, narrower gap this template does not
/// yet claim (the fixture's own four shapes are all arity-matched calls);
/// only a WRITTEN, MATCHED argument's own sort is judged.
pub fn judge_manifest_call(
    module_name: &str,
    entry: &ManifestEntry,
    positional: &[(AbstractValue, ruff_text_size::TextRange)],
    keyword: &[(String, AbstractValue, ruff_text_size::TextRange)],
    kernel: &Arc<RefinedTSKernel>,
) -> ManifestCallOutcome {
    let mut fires = Vec::new();
    let positional_parameters: Vec<&ManifestParameter> = entry.parameters.iter().filter(|p| !p.keyword_only).collect();
    for (parameter, (value, range)) in positional_parameters.iter().zip(positional.iter()) {
        judge_one_argument(module_name, entry, parameter, value, *range, kernel, &mut fires);
    }
    for (name, value, range) in keyword {
        if let Some(parameter) = entry.parameters.iter().find(|p| &p.name == name) {
            judge_one_argument(module_name, entry, parameter, value, *range, kernel, &mut fires);
        }
    }
    ManifestCallOutcome {
        fires,
        return_decline_sentence: diagnostic_sentences::manifest_entry_names_no_producer(
            module_name,
            &entry.function_name,
            &entry.producer_symbol,
        ),
    }
}

/// One argument's own crossing-fit judge, pushed onto `fires` on a proved
/// refutation — a kernel refusal (`Err(())`) contributes NOTHING (the
/// argument stays unjudged, never wrongly refuted), matching
/// `foreign_scalar_subset`'s own refusal posture.
fn judge_one_argument(
    module_name: &str,
    entry: &ManifestEntry,
    parameter: &ManifestParameter,
    value: &AbstractValue,
    range: ruff_text_size::TextRange,
    kernel: &Arc<RefinedTSKernel>,
    fires: &mut Vec<(ruff_text_size::TextRange, String)>,
) {
    if let Ok(false) = entry_crossing_fits(value, parameter.sort, kernel) {
        fires.push((
            range,
            diagnostic_sentences::manifest_entry_crossing_refused(
                module_name,
                &entry.function_name,
                &parameter.name,
                &crossing_value_words(value),
                parameter.sort.spelling(),
            ),
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use refined_domain::abstract_value::known_values;
    use refined_domain::abstract_value::unknown;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;

    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn zero_range() -> TextRange {
        TextRange::new(TextSize::from(0), TextSize::from(0))
    }

    /// The grammar's own basic shape: two positional parameters, one
    /// keyword-only with a default.
    #[test]
    fn parses_the_python_arg_parser_subset() {
        let parsed = parse_entry_contract("scale(Scalar value, int factor=1, *, bool clamp=False)")
            .expect("a well-formed signature must parse");
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].name, "value");
        assert_eq!(parsed[0].sort, ManifestSort::Scalar);
        assert!(!parsed[0].keyword_only);
        assert!(!parsed[0].has_default);
        assert_eq!(parsed[1].name, "factor");
        assert_eq!(parsed[1].sort, ManifestSort::Int);
        assert!(parsed[1].has_default);
        assert_eq!(parsed[2].name, "clamp");
        assert_eq!(parsed[2].sort, ManifestSort::Bool);
        assert!(parsed[2].keyword_only);
    }

    /// A zero-parameter signature parses to an empty list, not a decline.
    #[test]
    fn parses_a_zero_parameter_signature() {
        let parsed = parse_entry_contract("ping()").expect("an empty parameter list must parse");
        assert!(parsed.is_empty());
    }

    /// A signature naming an unrecognized Sort word declines the WHOLE
    /// contract, never a partial parameter list.
    #[test]
    fn an_unrecognized_sort_word_declines_the_whole_contract() {
        assert!(parse_entry_contract("scale(Tensor value)").is_none());
    }

    /// A parameter with no Sort word at all (bare `value`, no type)
    /// declines the whole contract.
    #[test]
    fn a_parameter_with_no_sort_word_declines_the_whole_contract() {
        assert!(parse_entry_contract("scale(value)").is_none());
    }

    /// A signature with no closing paren declines outright.
    #[test]
    fn an_unbalanced_signature_declines() {
        assert!(parse_entry_contract("scale(Scalar value").is_none());
    }

    /// An Integer-tagged value fits an `int`-sorted parameter; a
    /// Float-tagged value against the same parameter is a real
    /// refutation.
    #[test]
    fn judges_int_sorted_parameter_crossing() {
        let Some(kernel) = loaded_kernel() else { return };
        let entry = ManifestEntry {
            function_name: "scale".to_owned(),
            parameters: vec![ManifestParameter {
                name: "factor".to_owned(),
                sort: ManifestSort::Int,
                keyword_only: false,
                has_default: false,
            }],
            producer_symbol: "widgets_scale_impl".to_owned(),
        };
        let fitting = known_values(vec![2.0], PrimitiveKind::Integer, TrustProved);
        let outcome = judge_manifest_call("widgets", &entry, &[(fitting, zero_range())], &[], &kernel);
        assert!(outcome.fires.is_empty(), "an int argument must fit an int-sorted parameter: {:?}", outcome.fires);

        let escaping = known_values(vec![2.5], PrimitiveKind::Float, TrustProved);
        let outcome = judge_manifest_call("widgets", &entry, &[(escaping, zero_range())], &[], &kernel);
        assert_eq!(outcome.fires.len(), 1, "a float argument must escape an int-sorted parameter: {:?}", outcome.fires);
        assert!(outcome.fires[0].1.contains("widgets.scale"));
        assert!(outcome.fires[0].1.contains("'factor: int'"));
    }

    /// A str-sorted argument against an int-sorted parameter is a
    /// structural sort mismatch, refused outright.
    #[test]
    fn a_string_argument_escapes_a_numeric_parameter() {
        let Some(kernel) = loaded_kernel() else { return };
        let entry = ManifestEntry {
            function_name: "scale".to_owned(),
            parameters: vec![ManifestParameter {
                name: "factor".to_owned(),
                sort: ManifestSort::Int,
                keyword_only: false,
                has_default: false,
            }],
            producer_symbol: "widgets_scale_impl".to_owned(),
        };
        let string_value = known_values(vec![104.0], PrimitiveKind::String, TrustProved);
        let outcome = judge_manifest_call("widgets", &entry, &[(string_value, zero_range())], &[], &kernel);
        assert_eq!(outcome.fires.len(), 1, "{:?}", outcome.fires);
    }

    /// An UNKNOWN argument (a kernel refusal's own honest twin) is left
    /// unjudged — no fire, since the walk cannot yet read its sort at
    /// all.
    #[test]
    fn an_unknown_argument_is_left_unjudged() {
        let Some(kernel) = loaded_kernel() else { return };
        let entry = ManifestEntry {
            function_name: "scale".to_owned(),
            parameters: vec![ManifestParameter {
                name: "factor".to_owned(),
                sort: ManifestSort::Int,
                keyword_only: false,
                has_default: false,
            }],
            producer_symbol: "widgets_scale_impl".to_owned(),
        };
        let outcome = judge_manifest_call("widgets", &entry, &[(unknown(), zero_range())], &[], &kernel);
        assert!(outcome.fires.is_empty(), "an unknown argument must never be refused outright: {:?}", outcome.fires);
    }

    /// The return's own decline names the producer symbol regardless of
    /// whether any argument fired — the producer half is simply not
    /// built yet.
    #[test]
    fn the_return_always_names_the_missing_producer() {
        let Some(kernel) = loaded_kernel() else { return };
        let entry = ManifestEntry {
            function_name: "scale".to_owned(),
            parameters: vec![],
            producer_symbol: "widgets_scale_impl".to_owned(),
        };
        let outcome = judge_manifest_call("widgets", &entry, &[], &[], &kernel);
        assert!(outcome.return_decline_sentence.contains("widgets.scale"));
        assert!(outcome.return_decline_sentence.contains("widgets_scale_impl"));
    }

    /// A well-formed manifest file, discovered beside a fixture
    /// directory, reads back its function row with the parsed entry
    /// contract and producer symbol.
    #[test]
    fn discovers_and_reads_a_well_formed_manifest() {
        let root = std::env::temp_dir().join(format!(
            "refinedpy_binding_manifest_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&root).expect("create temp dir");
        let manifest_json = serde_json::json!({
            "scale": {"entry": "scale(Scalar value, int factor=1)", "producer": "widgets_scale_impl"},
        });
        fs::write(root.join("widgets.manifest.json"), manifest_json.to_string()).expect("write manifest");

        let discovered = discover_manifest("widgets", Some(&root)).expect("a manifest file exists");
        let manifest = discovered.expect("the manifest reads without error");
        let entry = manifest.entries.get("scale").expect("the scale row reads");
        assert_eq!(entry.producer_symbol, "widgets_scale_impl");
        assert_eq!(entry.parameters.len(), 2);

        fs::remove_dir_all(&root).ok();
    }

    /// No manifest file beside the directory at all answers `None` —
    /// rung 1's own plain decline territory, not an error this reader
    /// reports.
    #[test]
    fn no_manifest_file_answers_none() {
        let root = std::env::temp_dir().join(format!(
            "refinedpy_binding_manifest_test_missing_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&root).expect("create temp dir");
        assert!(discover_manifest("nonexistent_module", Some(&root)).is_none());
        fs::remove_dir_all(&root).ok();
    }

    /// A manifest file that IS present but is not readable JSON answers
    /// `Some(Err(sentence))`, naming the file.
    #[test]
    fn an_unreadable_manifest_file_names_itself() {
        let root = std::env::temp_dir().join(format!(
            "refinedpy_binding_manifest_test_unreadable_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&root).expect("create temp dir");
        fs::write(root.join("widgets.manifest.json"), b"not json at all").expect("write garbage");

        let discovered = discover_manifest("widgets", Some(&root)).expect("the file exists");
        let sentence = discovered.expect_err("garbage JSON must decline");
        assert!(sentence.contains("widgets.manifest.json"), "{sentence}");

        fs::remove_dir_all(&root).ok();
    }
}
