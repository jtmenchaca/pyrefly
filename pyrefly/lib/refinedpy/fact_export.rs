/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The Python fact-export surface: one module's CHECKED facts written as
//! a JSON artifact another language's checker reads across the FFI edge
//! (CROSS-LANGUAGE-EDGE.md §8 — "the edge's return fact — the Python
//! function's kernel summary pushed through the return transport — IS
//! the fact on that expression").
//!
//! What crosses is what the checker DERIVED, never what an annotation
//! claimed: `check::derived_return_values` runs the same walk
//! `findings_for_module_with_resolver` runs (parameters seeded from
//! their declarations by `seed_parameters`, the body walked statement by
//! statement) and hands back the join of every value the body's
//! `return`s produced. The entry sets are the declared refinements the
//! walk itself seeds from (`typereading::declared_refinement`), so a
//! consumer reading the entry and the return reads exactly the two ends
//! of one derivation.
//!
//! EVERY FIELD IS COMPUTED. A def with a parameter carrying no declared
//! refinement, or with a derived return that has no faithful set reading
//! (an object, an unknown), is OMITTED from the artifact with the reason
//! named on stderr — the artifact never carries a stub, a placeholder,
//! or a widened stand-in for a fact this checker did not derive.
//!
//! The premises the artifact's own fields discharge (§5, "Edge
//! premises"):
//!
//! - TARGET INTEGRITY — `target.contentHash` is sha256 over the file's
//!   exact bytes, so a consumer can check that the code that runs is the
//!   code that was checked.
//! - RUNTIME IDENTITY — `runtime.band` states the semantics the Python
//!   pins commit to, which the derived facts inherit.
//! - CHANNEL PURITY — `return.stdoutPure` is the effect fact §5 names:
//!   the target writes nothing else to the channel the wire uses.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::lattice_operations::set_of_known;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::wire_format::wire_set;
use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::CmpOp;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;

use crate::refinedpy::check::derived_return_values;
use crate::refinedpy::cross_module::ModuleResolver;
use crate::refinedpy::env::Environment;
use crate::refinedpy::surface::SurfaceImports;
use crate::refinedpy::surface::compile_aliases;
use crate::refinedpy::surface::surface_imports;
use crate::refinedpy::typereading::DeclaredRefinement;
use crate::refinedpy::typereading::base_sort_return_refinement;
use crate::refinedpy::typereading::declared_refinement;

/// The artifact's own kind tag and version — the Go consumer matches on
/// both before reading a single fact.
const ARTIFACT_KIND: &str = "python-fact-artifact";
const ARTIFACT_VERSION: i64 = 1;

/// The semantics band every fact in this artifact inherits: the Python
/// pins commit to CPython 3.11+ behaviour, not to "Python"
/// (CROSS-LANGUAGE-EDGE.md §5, "Runtime identity").
const RUNTIME_BAND: &str = "cpython-3.11+";

/// One def this export could not carry, and the construct that stopped
/// it — the work-queue row a reader turns into a fix.
pub struct Omission {
    pub function: String,
    pub reason: String,
}

/// One module's export: the artifact, and every def omitted from it.
pub struct Export {
    pub artifact: Value,
    pub omissions: Vec<Omission>,
}

/// `module`'s fact artifact — every top-level `def` with a fully
/// declared refined entry and a derivable return set, in source order.
///
/// `source_bytes` is the target file's exact content (the bytes the hash
/// commits to, and the bytes `module` was parsed from); `basename` is
/// what `target.file` states. `resolver` is the same import resolver the
/// checker's own CLI passes, so a def reading an imported name derives
/// exactly what the checker derives for it.
pub fn export_module(
    module: &ModModule,
    source_bytes: &[u8],
    basename: &str,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
) -> Export {
    let aliases = compile_aliases(module);
    let imports = surface_imports(module);
    // ONE walk of the whole module answers every def's derived return:
    // the shared context (imports resolved, function/class tables built)
    // costs the same whether one def asks or all of them do.
    let derived_returns = derived_return_values(module, resolver, kernel);
    let mut functions = Map::new();
    let mut omissions = Vec::new();
    let module_line_starts = line_starts_of(source_bytes);

    for def in top_level_defs(module) {
        let name = def.name.id.as_str().to_owned();
        match export_function(def, module, &module_line_starts, &aliases, &imports, &derived_returns) {
            Ok(entry) => {
                functions.insert(name, entry);
            }
            Err(reason) => omissions.push(Omission { function: name, reason }),
        }
    }

    let mut artifact = Map::new();
    artifact.insert(
        "refined".to_owned(),
        json!({"kind": ARTIFACT_KIND, "version": ARTIFACT_VERSION}),
    );
    artifact.insert(
        "target".to_owned(),
        json!({"file": basename, "contentHash": format!("sha256:{}", sha256_hex(source_bytes))}),
    );
    artifact.insert("runtime".to_owned(), json!({"band": RUNTIME_BAND}));
    if let Some(called) = harness_call(module) {
        // A harness is exported only when the module's `__main__` block
        // IS the stdin-JSON/stdout-JSON shape; absence of the key is the
        // consumer's "no harness fact" (§11's own reading), never a
        // guessed default.
        artifact.insert(
            "harness".to_owned(),
            json!({"stdin": "json", "stdout": "json", "calls": called}),
        );
    }
    artifact.insert("functions".to_owned(), Value::Object(functions));

    Export {
        artifact: Value::Object(artifact),
        omissions,
    }
}

/// Every top-level `def` in `module`, in source order.
fn top_level_defs(module: &ModModule) -> impl Iterator<Item = &StmtFunctionDef> {
    module.body.iter().filter_map(|stmt| match stmt {
        Stmt::FunctionDef(def) => Some(def),
        _ => None,
    })
}

/// One def's exported entry, or the reason it cannot be exported.
///
/// Every field is derived here and nothing is defaulted: the entry from
/// the declarations the walk itself seeds from, the return from the walk
/// itself, `stdoutPure` from a scan of the body and the same-module defs
/// it calls, and `said` from the same diagnostics formatter every
/// refinement sentence in this checker is spelled through.
fn export_function(
    def: &StmtFunctionDef,
    module: &ModModule,
    line_starts: &[usize],
    aliases: &HashMap<String, RefinedSet>,
    imports: &SurfaceImports,
    derived_returns: &HashMap<String, AbstractValue>,
) -> Result<Value, String> {
    let entry = entry_rows(def, aliases, imports)?;
    let name = def.name.id.as_str();
    let returned = derived_returns
        .get(name)
        .ok_or_else(|| "the body's returns derived no value the walk could read".to_owned())?;
    let return_set = faithful_return_set(returned)?;
    let stdout_pure = writes_nothing_to_stdout(def, module);
    // The def's own NAME identifier, not the statement range: a
    // decorated def's statement range starts at the decorator, and the
    // line a reader means by "the def line" is the one `def <name>` sits
    // on. The name identifier is always on that line.
    let line = line_of(line_starts, def.name.range.start().to_usize());
    let said = provenance_sentence(&entry, &return_set);

    let entry_json: Vec<Value> = entry.iter().map(entry_row_json).collect();
    Ok(json!({
        "entry": entry_json,
        "return": {"set": wire_set(&return_set), "stdoutPure": stdout_pure},
        "provenance": {"line": line, "said": said},
    }))
}

/// One exported parameter: its name, and the shape its declaration
/// states — a SEQUENCE (an element set plus the length floor its own
/// declaration carries) or a SCALAR (one set).
struct EntryRow {
    name: String,
    shape: EntryShape,
}

enum EntryShape {
    /// `list[X]`/`set[X]`/`Sequence[X]` — the element's own set, and the
    /// declaration's own length floor (`element_length`'s `lo`, 0 when
    /// the declaration states no bound, exactly the default
    /// `check::seed_parameters` seeds the repetition window with).
    Sequence {
        element: RefinedSet,
        length_at_least: i64,
    },
    /// Every other readable declaration: the set itself.
    Scalar(RefinedSet),
}

/// `def`'s parameters read into exported entry rows, or the reason the
/// whole def cannot be exported.
///
/// Reads each parameter through the SAME two-step `seed_parameters`
/// takes — `declared_refinement`, falling back to the bare
/// `int`/`float`/`str` sort reading — so a parameter the walk seeds is a
/// parameter this export can state, and a parameter the walk leaves
/// unseeded omits the def rather than crossing an unfounded set.
fn entry_rows(
    def: &StmtFunctionDef,
    aliases: &HashMap<String, RefinedSet>,
    imports: &SurfaceImports,
) -> Result<Vec<EntryRow>, String> {
    // A variadic tail has no fixed arity and therefore no entry row a
    // caller on the other side of the wire could fill — checked before
    // any row is read, so the reason names the real obstacle.
    if def.parameters.vararg.is_some() || def.parameters.kwarg.is_some() {
        return Err("a '*args'/'**kwargs' parameter states no crossable entry shape".to_owned());
    }
    // The same no-locals environment every module-level annotation read
    // in this checker uses: a top-level def's own annotation is never
    // shadowed by a body-local rebinding.
    let environment = Environment::new(HashSet::new());
    let mut rows = Vec::new();
    for parameter in def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .chain(def.parameters.kwonlyargs.iter())
    {
        let parameter_name = parameter.parameter.name.id.as_str().to_owned();
        let Some(annotation) = parameter.parameter.annotation.as_deref() else {
            return Err(format!("parameter '{parameter_name}' carries no annotation"));
        };
        let Some(declared) = declared_refinement(annotation, aliases, imports, &environment)
            .or_else(|| base_sort_return_refinement(annotation))
        else {
            return Err(format!(
                "parameter '{parameter_name}' carries no refinement this checker reads"
            ));
        };
        rows.push(EntryRow {
            name: parameter_name,
            shape: entry_shape(&declared)?,
        });
    }
    if rows.is_empty() {
        return Err("the def declares no parameters, so it states no refined entry".to_owned());
    }
    Ok(rows)
}

/// One declaration read into the shape this artifact carries — the same
/// SEQUENCE/SCALAR split `check::seed_parameters` makes when it seeds
/// the parameter (a container spelling with a non-empty element set
/// seeds a repetition window; everything else seeds the set itself).
fn entry_shape(declared: &DeclaredRefinement) -> Result<EntryShape, String> {
    let is_sequence_container = declared.spelling.starts_with("list[")
        || declared.spelling.starts_with("set[")
        || declared.spelling.starts_with("Sequence[");
    if is_sequence_container {
        let Some(element) = declared.element.as_ref() else {
            return Err(format!(
                "'{}' states a container with no element refinement",
                declared.spelling
            ));
        };
        if element.set.forms.is_empty() {
            return Err(format!(
                "'{}' states a container whose element is itself a container, which crosses no set",
                declared.spelling
            ));
        }
        let (lo, _hi) = declared.element_length.unwrap_or((0, None));
        return Ok(EntryShape::Sequence {
            element: element.set.clone(),
            length_at_least: lo,
        });
    }
    if declared.set.forms.is_empty() {
        return Err(format!(
            "'{}' states a shape (a dict, a tuple, a TypedDict, a generator) that crosses no single set",
            declared.spelling
        ));
    }
    Ok(EntryShape::Scalar(declared.set.clone()))
}

/// One entry row as the artifact spells it.
fn entry_row_json(row: &EntryRow) -> Value {
    match &row.shape {
        EntryShape::Sequence {
            element,
            length_at_least,
        } => json!({
            "name": row.name,
            "sequence": {"element": wire_set(element), "lengthAtLeast": length_at_least},
        }),
        EntryShape::Scalar(set) => json!({"name": row.name, "set": wire_set(set)}),
    }
}

/// The derived return read as a set, or the reason it has no faithful
/// reading. `set_of_known` is the one converter — an object, an unknown,
/// a nested sequence answers `None` there, and this states which.
fn faithful_return_set(
    returned: &AbstractValue,
) -> Result<RefinedSet, String> {
    if let Some(set) = set_of_known(returned) {
        if set.forms.is_empty() {
            return Err("the derived return is the empty set, which states no crossable fact".to_owned());
        }
        return Ok(set);
    }
    Err(format!(
        "the derived return is {}, which has no faithful set reading",
        return_kind_words(returned)
    ))
}

/// Plain words for what a return derived to, for the omission row.
fn return_kind_words(returned: &AbstractValue) -> &'static str {
    // a set-kinded value only reaches here when it wears a sort tag —
    // `set_of_known` answers Some for an untagged one
    if returned.kind == Kind::Set {
        return "a set of values whose members are not plain numbers";
    }
    match returned.kind {
        Kind::Unknown => "a value this walk never determined",
        Kind::Object | Kind::ObjectStar => "an object",
        Kind::List | Kind::Collection => "a nested sequence",
        Kind::Promise => "an awaitable",
        Kind::Date => "a date",
        Kind::Symbol => "a symbol",
        Kind::HostFunction => "a function",
        Kind::Bigints => "an arbitrary-width integer",
        Kind::Regex => "a regular expression",
        Kind::Undef | Kind::Null => "the absent value",
        Kind::NaN => "NaN",
        Kind::PossiblyUndefined => "a possibly-absent value",
        Kind::PossiblyNaN => "a possibly-NaN value",
        Kind::KindUnion => "a union of sorts",
        Kind::ArrayHoles => "a sequence of holes",
        // the empty tuple is the one Values shape with no set spelling
        Kind::Values => "the empty tuple",
        // handled above, and answered by set_of_known otherwise
        Kind::Set | Kind::Variable => "a value with no set reading",
    }
}

/// The one sentence `provenance.said` states, assembled from the facts
/// the artifact already carries — each entry bound and the derived
/// return, spelled through `format_for_diagnostics`, the same formatter
/// every refinement sentence in this checker is spelled through.
fn provenance_sentence(
    entry: &[EntryRow],
    return_set: &RefinedSet,
) -> String {
    let entry_words: Vec<String> = entry
        .iter()
        .map(|row| match &row.shape {
            EntryShape::Sequence {
                element,
                length_at_least,
            } => format!(
                "'{}' whose every element is {} and whose length is at least {}",
                row.name,
                format_for_diagnostics(element),
                length_at_least
            ),
            EntryShape::Scalar(set) => {
                format!("'{}' is {}", row.name, format_for_diagnostics(set))
            }
        })
        .collect();
    format!(
        "given {}, this body's returns derive {}",
        entry_words.join(" and "),
        format_for_diagnostics(return_set)
    )
}

// --- CHANNEL PURITY --------------------------------------------------

/// Whether `def`'s body writes nothing to stdout — the effect fact
/// CROSS-LANGUAGE-EDGE.md §5 ("Channel purity") names, and the premise
/// the wire's own claim rests on: the wire IS stdout, so a stray write
/// inside the target corrupts the payload.
///
/// A CONSERVATIVE syntactic scan: any `print(...)` call, any
/// `sys.stdout.<anything>(...)` / `sys.stdout.write` reference, and any
/// `.write(...)` on a receiver spelled `stdout` counts as a write.
/// Transitive through the SAME-MODULE defs the body calls, capped at the
/// module (a call this module does not declare — an import, a builtin
/// this scan does not model — makes the answer false, since the scan
/// cannot see that body).
fn writes_nothing_to_stdout(def: &StmtFunctionDef, module: &ModModule) -> bool {
    let module_defs: HashMap<&str, &StmtFunctionDef> = top_level_defs(module)
        .map(|candidate| (candidate.name.id.as_str(), candidate))
        .collect();
    let mut visited: HashSet<String> = HashSet::new();
    body_is_stdout_pure(&def.body, &module_defs, &mut visited)
}

/// One body's scan, following every same-module call it makes.
/// `visited` names the defs already scanned, so a recursive or mutually
/// recursive call terminates (a def already being scanned adds nothing
/// new to the answer).
fn body_is_stdout_pure(
    body: &[Stmt],
    module_defs: &HashMap<&str, &StmtFunctionDef>,
    visited: &mut HashSet<String>,
) -> bool {
    // The scan collects first and decides second: the decision recurses
    // into a callee's own body, which would otherwise need `visited`
    // borrowed inside the traversal closure that already borrows it.
    let mut writes_stdout = false;
    let mut called_names: Vec<String> = Vec::new();
    let mut has_opaque_call = false;
    for stmt in body {
        walk_statement_expressions(stmt, &mut |expr| {
            if expression_writes_stdout(expr) {
                writes_stdout = true;
                return;
            }
            let Expr::Call(call) = expr else {
                return;
            };
            match call.func.as_ref() {
                Expr::Name(callee) => called_names.push(callee.id.as_str().to_owned()),
                // an attribute call (`obj.method(...)`, `math.sqrt(...)`)
                // reaches no same-module def this scan can follow; the
                // stdout-writing attribute shapes are already caught by
                // `expression_writes_stdout` above, and a method body on
                // an instance is out of this scan's reach — so any
                // receiver outside the modelled stdlib list refuses the
                // claim.
                other => {
                    if is_opaque_receiver_call(other) {
                        has_opaque_call = true;
                    }
                }
            }
        });
    }
    if writes_stdout || has_opaque_call {
        return false;
    }
    for name in called_names {
        if is_pure_builtin(&name) {
            continue;
        }
        let Some(callee_def) = module_defs.get(name.as_str()) else {
            // a name this module does not declare: an import, or a
            // builtin outside the modelled list. The scan cannot see
            // that body, so it cannot claim the channel is clean.
            return false;
        };
        // a def already being scanned adds nothing new to the answer,
        // which is what makes a recursive or mutually recursive call
        // terminate here
        if !visited.insert(name) {
            continue;
        }
        if !body_is_stdout_pure(&callee_def.body, module_defs, visited) {
            return false;
        }
    }
    true
}

/// Whether `expr` is itself a write to stdout: `print(...)`, a
/// `sys.stdout` attribute path, or a `.write(...)` whose receiver is
/// spelled `stdout`.
fn expression_writes_stdout(expr: &Expr) -> bool {
    match expr {
        // `print(...)`, or any call whose receiver path names stdout
        // (`sys.stdout.write(...)`, `sys.stdout.flush()`).
        Expr::Call(call) => match call.func.as_ref() {
            Expr::Name(callee) => callee.id.as_str() == "print",
            Expr::Attribute(attribute) => attribute_path_reaches_stdout(attribute.value.as_ref()),
            _ => false,
        },
        // A bare `sys.stdout` reference (handed to a writer this scan
        // cannot follow) is itself enough to refuse the claim.
        Expr::Attribute(_) => attribute_path_reaches_stdout(expr),
        _ => false,
    }
}

/// Whether an attribute path's own spelling names stdout — `sys.stdout`,
/// or a bare `stdout` a `from sys import stdout` would bind.
fn attribute_path_reaches_stdout(expr: &Expr) -> bool {
    match expr {
        Expr::Name(name) => name.id.as_str() == "stdout",
        Expr::Attribute(attribute) => {
            attribute.attr.as_str() == "stdout" || attribute_path_reaches_stdout(attribute.value.as_ref())
        }
        _ => false,
    }
}

/// Whether an attribute-callee shape is one this scan cannot follow and
/// therefore refuses the purity claim for. A `math.<fn>(...)` /
/// `json.<fn>(...)` call on a stdlib module this checker already models
/// writes nothing to the channel; every other receiver (an instance
/// method, an imported module's function) is opaque here.
fn is_opaque_receiver_call(func: &Expr) -> bool {
    let Expr::Attribute(attribute) = func else {
        return true;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return true;
    };
    !matches!(receiver.id.as_str(), "math" | "json")
}

/// The builtins this scan knows write nothing to stdout. A name outside
/// this list and outside the module's own defs refuses the claim, so the
/// list only ever ADMITS a fact; it never widens one.
fn is_pure_builtin(name: &str) -> bool {
    matches!(
        name,
        "abs" | "all"
            | "any"
            | "bool"
            | "dict"
            | "divmod"
            | "enumerate"
            | "float"
            | "int"
            | "len"
            | "list"
            | "max"
            | "min"
            | "pow"
            | "range"
            | "round"
            | "set"
            | "sorted"
            | "str"
            | "sum"
            | "tuple"
            | "zip"
    )
}

// --- THE HARNESS -----------------------------------------------------

/// The function a module's `if __name__ == "__main__":` block calls,
/// when that block IS the stdin-JSON/stdout-JSON shape §11 names:
/// `print(json.dumps(<f>(json.load(sys.stdin))))`. `None` for a module
/// with no main block, or with a block of any other shape — the artifact
/// omits the harness key entirely then, and the consumer reads that
/// absence as "no harness fact".
fn harness_call(module: &ModModule) -> Option<String> {
    for stmt in &module.body {
        let Stmt::If(if_stmt) = stmt else {
            continue;
        };
        if !is_main_guard(if_stmt.test.as_ref()) {
            continue;
        }
        for inner in &if_stmt.body {
            let Stmt::Expr(expr_stmt) = inner else {
                continue;
            };
            if let Some(called) = harness_shape_call(expr_stmt.value.as_ref()) {
                return Some(called);
            }
        }
    }
    None
}

/// Whether `test` is `__name__ == "__main__"` (either order).
fn is_main_guard(test: &Expr) -> bool {
    let Expr::Compare(compare) = test else {
        return false;
    };
    let ([CmpOp::Eq], [right]) = (compare.ops.as_ref(), compare.comparators.as_ref()) else {
        return false;
    };
    let names_dunder_name = |expr: &Expr| matches!(expr, Expr::Name(name) if name.id.as_str() == "__name__");
    let names_main = |expr: &Expr| {
        matches!(expr, Expr::StringLiteral(literal) if literal.value.to_str() == "__main__")
    };
    (names_dunder_name(compare.left.as_ref()) && names_main(right))
        || (names_main(compare.left.as_ref()) && names_dunder_name(right))
}

/// `print(json.dumps(<f>(json.load(sys.stdin))))` read for its `<f>`.
/// Every layer must match: a bare `print` call of one argument, a
/// `json.dumps` call of one argument, a bare-Name call of one argument,
/// and a `json.load(sys.stdin)` innermost. Any deviation answers `None`
/// — a harness this reader half-recognizes is not a harness fact.
fn harness_shape_call(expr: &Expr) -> Option<String> {
    let printed = single_argument_of(expr, &CalleeSpelling::BareName("print"))?;
    let dumped = single_argument_of(printed, &CalleeSpelling::Attribute("json", "dumps"))?;
    let Expr::Call(inner) = dumped else {
        return None;
    };
    let Expr::Name(called) = inner.func.as_ref() else {
        return None;
    };
    if !inner.arguments.keywords.is_empty() {
        return None;
    }
    let [loaded] = inner.arguments.args.as_ref() else {
        return None;
    };
    let stdin = single_argument_of(loaded, &CalleeSpelling::Attribute("json", "load"))?;
    if !is_sys_stdin(stdin) {
        return None;
    }
    Some(called.id.as_str().to_owned())
}

/// How a harness layer's callee must be spelled.
enum CalleeSpelling {
    BareName(&'static str),
    Attribute(&'static str, &'static str),
}

/// `expr`'s single positional argument, when `expr` is a call to the
/// named callee with exactly one positional argument and no keywords.
fn single_argument_of<'a>(expr: &'a Expr, callee: &CalleeSpelling) -> Option<&'a Expr> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let matches_callee = match (callee, call.func.as_ref()) {
        (CalleeSpelling::BareName(wanted), Expr::Name(name)) => name.id.as_str() == *wanted,
        (CalleeSpelling::Attribute(module, attribute), Expr::Attribute(path)) => {
            path.attr.as_str() == *attribute
                && matches!(path.value.as_ref(), Expr::Name(receiver) if receiver.id.as_str() == *module)
        }
        _ => false,
    };
    if !matches_callee || !call.arguments.keywords.is_empty() {
        return None;
    }
    let [only] = call.arguments.args.as_ref() else {
        return None;
    };
    Some(only)
}

/// Whether `expr` is `sys.stdin` (or a bare `stdin` a `from sys import
/// stdin` would bind).
fn is_sys_stdin(expr: &Expr) -> bool {
    match expr {
        Expr::Name(name) => name.id.as_str() == "stdin",
        Expr::Attribute(attribute) => {
            attribute.attr.as_str() == "stdin"
                && matches!(attribute.value.as_ref(), Expr::Name(receiver) if receiver.id.as_str() == "sys")
        }
        _ => false,
    }
}

// --- POSITIONS -------------------------------------------------------

/// Every line's own start offset in `source`, so a byte offset reads
/// back as a 1-based line number.
fn line_starts_of(source: &[u8]) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (index, byte) in source.iter().enumerate() {
        if *byte == b'\n' {
            starts.push(index + 1);
        }
    }
    starts
}

/// The 1-based line `offset` sits on.
fn line_of(line_starts: &[usize], offset: usize) -> i64 {
    match line_starts.binary_search(&offset) {
        Ok(index) => (index + 1) as i64,
        Err(index) => index as i64,
    }
}

// --- STATEMENT/EXPRESSION TRAVERSAL ----------------------------------

/// Every expression inside one statement, including every nested
/// statement's own — the traversal the stdout scan walks. Visits each
/// expression node once, parents before children.
fn walk_statement_expressions(stmt: &Stmt, visit: &mut dyn FnMut(&Expr)) {
    match stmt {
        Stmt::Expr(node) => walk_expression(node.value.as_ref(), visit),
        Stmt::Return(node) => {
            if let Some(value) = node.value.as_deref() {
                walk_expression(value, visit);
            }
        }
        Stmt::Assign(node) => {
            for target in &node.targets {
                walk_expression(target, visit);
            }
            walk_expression(node.value.as_ref(), visit);
        }
        Stmt::AnnAssign(node) => {
            if let Some(value) = node.value.as_deref() {
                walk_expression(value, visit);
            }
        }
        Stmt::AugAssign(node) => walk_expression(node.value.as_ref(), visit),
        Stmt::If(node) => {
            walk_expression(node.test.as_ref(), visit);
            for inner in &node.body {
                walk_statement_expressions(inner, visit);
            }
            for clause in &node.elif_else_clauses {
                if let Some(test) = clause.test.as_ref() {
                    walk_expression(test, visit);
                }
                for inner in &clause.body {
                    walk_statement_expressions(inner, visit);
                }
            }
        }
        Stmt::For(node) => {
            walk_expression(node.iter.as_ref(), visit);
            for inner in node.body.iter().chain(node.orelse.iter()) {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::While(node) => {
            walk_expression(node.test.as_ref(), visit);
            for inner in node.body.iter().chain(node.orelse.iter()) {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::With(node) => {
            for item in &node.items {
                walk_expression(&item.context_expr, visit);
            }
            for inner in &node.body {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::Try(node) => {
            for inner in node.body.iter().chain(node.orelse.iter()).chain(node.finalbody.iter()) {
                walk_statement_expressions(inner, visit);
            }
            for handler in &node.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                for inner in &handler.body {
                    walk_statement_expressions(inner, visit);
                }
            }
        }
        Stmt::Match(node) => {
            walk_expression(node.subject.as_ref(), visit);
            for case in &node.cases {
                for inner in &case.body {
                    walk_statement_expressions(inner, visit);
                }
            }
        }
        Stmt::FunctionDef(node) => {
            for inner in &node.body {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::ClassDef(node) => {
            for inner in &node.body {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::Raise(node) => {
            if let Some(exception) = node.exc.as_deref() {
                walk_expression(exception, visit);
            }
            if let Some(cause) = node.cause.as_deref() {
                walk_expression(cause, visit);
            }
        }
        Stmt::Assert(node) => {
            walk_expression(node.test.as_ref(), visit);
            if let Some(message) = node.msg.as_deref() {
                walk_expression(message, visit);
            }
        }
        Stmt::Delete(node) => {
            for target in &node.targets {
                walk_expression(target, visit);
            }
        }
        // Every remaining form (`pass`, `break`, `continue`, `global`,
        // `nonlocal`, `import`, `from ... import`, a type alias) holds no
        // expression a stdout write could hide in.
        _ => {}
    }
}

/// One expression and every expression nested inside it, parent first.
fn walk_expression(expr: &Expr, visit: &mut dyn FnMut(&Expr)) {
    visit(expr);
    match expr {
        Expr::Call(node) => {
            walk_expression(node.func.as_ref(), visit);
            for argument in node.arguments.args.iter() {
                walk_expression(argument, visit);
            }
            for keyword in node.arguments.keywords.iter() {
                walk_expression(&keyword.value, visit);
            }
        }
        Expr::Attribute(node) => walk_expression(node.value.as_ref(), visit),
        Expr::Subscript(node) => {
            walk_expression(node.value.as_ref(), visit);
            walk_expression(node.slice.as_ref(), visit);
        }
        Expr::BinOp(node) => {
            walk_expression(node.left.as_ref(), visit);
            walk_expression(node.right.as_ref(), visit);
        }
        Expr::UnaryOp(node) => walk_expression(node.operand.as_ref(), visit),
        Expr::BoolOp(node) => {
            for value in &node.values {
                walk_expression(value, visit);
            }
        }
        Expr::Compare(node) => {
            walk_expression(node.left.as_ref(), visit);
            for comparator in node.comparators.iter() {
                walk_expression(comparator, visit);
            }
        }
        Expr::If(node) => {
            walk_expression(node.test.as_ref(), visit);
            walk_expression(node.body.as_ref(), visit);
            walk_expression(node.orelse.as_ref(), visit);
        }
        Expr::Tuple(node) => {
            for element in &node.elts {
                walk_expression(element, visit);
            }
        }
        Expr::List(node) => {
            for element in &node.elts {
                walk_expression(element, visit);
            }
        }
        Expr::Set(node) => {
            for element in &node.elts {
                walk_expression(element, visit);
            }
        }
        Expr::Dict(node) => {
            for item in &node.items {
                if let Some(key) = item.key.as_ref() {
                    walk_expression(key, visit);
                }
                walk_expression(&item.value, visit);
            }
        }
        Expr::ListComp(node) => {
            walk_expression(node.elt.as_ref(), visit);
            for generator in &node.generators {
                walk_expression(&generator.iter, visit);
                for condition in &generator.ifs {
                    walk_expression(condition, visit);
                }
            }
        }
        Expr::SetComp(node) => {
            walk_expression(node.elt.as_ref(), visit);
            for generator in &node.generators {
                walk_expression(&generator.iter, visit);
                for condition in &generator.ifs {
                    walk_expression(condition, visit);
                }
            }
        }
        Expr::Generator(node) => {
            walk_expression(node.elt.as_ref(), visit);
            for generator in &node.generators {
                walk_expression(&generator.iter, visit);
                for condition in &generator.ifs {
                    walk_expression(condition, visit);
                }
            }
        }
        Expr::DictComp(node) => {
            if let Some(key) = node.key.as_deref() {
                walk_expression(key, visit);
            }
            walk_expression(node.value.as_ref(), visit);
            for generator in &node.generators {
                walk_expression(&generator.iter, visit);
                for condition in &generator.ifs {
                    walk_expression(condition, visit);
                }
            }
        }
        Expr::Starred(node) => walk_expression(node.value.as_ref(), visit),
        Expr::Await(node) => walk_expression(node.value.as_ref(), visit),
        Expr::Yield(node) => {
            if let Some(value) = node.value.as_deref() {
                walk_expression(value, visit);
            }
        }
        Expr::YieldFrom(node) => walk_expression(node.value.as_ref(), visit),
        Expr::Named(node) => {
            walk_expression(node.target.as_ref(), visit);
            walk_expression(node.value.as_ref(), visit);
        }
        Expr::Lambda(node) => walk_expression(node.body.as_ref(), visit),
        Expr::Slice(node) => {
            for part in [node.lower.as_deref(), node.upper.as_deref(), node.step.as_deref()]
                .into_iter()
                .flatten()
            {
                walk_expression(part, visit);
            }
        }
        Expr::FString(node) => {
            for element in node.value.elements().filter_map(|element| element.as_interpolation()) {
                walk_expression(element.expression.as_ref(), visit);
            }
        }
        // Every remaining form is a leaf (a name, a literal, an
        // ellipsis) with nothing nested inside it.
        _ => {}
    }
}

// --- SHA-256 ---------------------------------------------------------
//
// FIPS 180-4 §6.2, implemented here rather than taken as a dependency:
// this crate's `Cargo.toml` is autocargo-generated from the Meta build
// definition, so a hand-added direct dependency does not survive a
// regeneration. The digest is exercised by this module's own tests
// against the standard published vectors.

/// The first 32 bits of the fractional parts of the cube roots of the
/// first 64 primes (FIPS 180-4 §4.2.2).
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// The first 32 bits of the fractional parts of the square roots of the
/// first 8 primes (FIPS 180-4 §5.3.3).
const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// `bytes`' SHA-256 digest as lowercase hex (FIPS 180-4 §6.2).
///
/// `pub(crate)`: `foreign_edge_artifact.rs`'s target-integrity check
/// reuses this exact digest rather than hand-rolling a second one, so a
/// hash computed on the export side and a hash computed on the read
/// side are provably the same function.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut state = SHA256_H0;
    // The padded message: the bytes, a 0x80 byte, zeros to 56 mod 64,
    // then the bit length as a big-endian u64.
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    padded.extend_from_slice(&bit_length.to_be_bytes());

    for block in padded.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (index, word) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(schedule[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut hex = String::with_capacity(64);
    for word in state {
        hex.push_str(&format!("{word:08x}"));
    }
    hex
}

#[cfg(test)]
mod tests {
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;

    use super::*;

    /// The dylib-absence convention every kernel-touching test in this
    /// crate follows (`lattice_conformance.rs`'s own helper): a missing
    /// artifact prints to stderr and the caller returns early, never
    /// failing the run.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    /// FIPS 180-4's own published vectors, plus the empty message.
    #[test]
    fn sha256_matches_the_published_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // one block longer than the single-block path, exercising the
        // multi-block loop
        assert_eq!(
            sha256_hex(&[b'a'; 1000]),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    /// The main-block reader recognizes the exact stdin→f→stdout shape
    /// and nothing looser.
    #[test]
    fn the_harness_reader_recognizes_the_json_stdio_shape() {
        let module = ruff_python_parser::parse_module(
            "import json\nimport sys\n\n\ndef f(x): return x\n\n\nif __name__ == \"__main__\":\n    print(json.dumps(f(json.load(sys.stdin))))\n",
        )
        .expect("test module parses")
        .into_syntax();
        assert_eq!(harness_call(&module).as_deref(), Some("f"));
    }

    #[test]
    fn a_main_block_of_another_shape_states_no_harness() {
        let module = ruff_python_parser::parse_module(
            "import json\nimport sys\n\n\ndef f(x): return x\n\n\nif __name__ == \"__main__\":\n    print(f(1))\n",
        )
        .expect("test module parses")
        .into_syntax();
        assert!(harness_call(&module).is_none());

        let no_main = ruff_python_parser::parse_module("def f(x): return x\n")
            .expect("test module parses")
            .into_syntax();
        assert!(harness_call(&no_main).is_none());
    }

    /// A `print` anywhere in the body — or in a same-module def the body
    /// calls — refuses the channel-purity claim.
    #[test]
    fn the_stdout_scan_follows_same_module_calls() {
        let module = ruff_python_parser::parse_module(
            "def quiet(x):\n    return x + 1\n\n\ndef loud(x):\n    print(x)\n    return x\n\n\ndef calls_quiet(x):\n    return quiet(x)\n\n\ndef calls_loud(x):\n    return loud(x)\n",
        )
        .expect("test module parses")
        .into_syntax();
        let defs: Vec<&StmtFunctionDef> = top_level_defs(&module).collect();
        let by_name = |wanted: &str| {
            *defs
                .iter()
                .find(|def| def.name.id.as_str() == wanted)
                .expect("the test module declares this def")
        };
        assert!(writes_nothing_to_stdout(by_name("quiet"), &module));
        assert!(!writes_nothing_to_stdout(by_name("loud"), &module));
        assert!(writes_nothing_to_stdout(by_name("calls_quiet"), &module));
        assert!(!writes_nothing_to_stdout(by_name("calls_loud"), &module));
    }

    /// A `sys.stdout.write(...)` is the same refusal a `print` is.
    #[test]
    fn the_stdout_scan_catches_a_direct_stdout_write() {
        let module = ruff_python_parser::parse_module(
            "import sys\n\n\ndef writes(x):\n    sys.stdout.write(\"hi\")\n    return x\n",
        )
        .expect("test module parses")
        .into_syntax();
        let def = top_level_defs(&module).next().expect("one def");
        assert!(!writes_nothing_to_stdout(def, &module));
    }

    /// The tutorial fixture exported end to end: every artifact key
    /// present, the hash prefixed and full-width, and the entry row
    /// carrying the sequence shape `samples`' own declaration states.
    /// Pinned at the STRUCTURE, not at the set contents — what the
    /// checker derives for the return is the derivation lanes' business,
    /// and this test must not restate it.
    #[test]
    fn the_tutorial_fixture_exports_its_structure() {
        let Some(kernel) = loaded_kernel() else {
            return;
        };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/tutorial/audio_level_python_only.py"
        );
        let source = std::fs::read(path).expect("the tutorial fixture is committed beside the checker");
        let text = String::from_utf8(source.clone()).expect("the fixture is UTF-8");
        let module = ruff_python_parser::parse_module(&text)
            .expect("the fixture parses")
            .into_syntax();
        let no_imports: ModuleResolver = &|_: &str| None;

        let export = export_module(
            &module,
            &source,
            "audio_level_python_only.py",
            no_imports,
            &kernel,
        );
        let artifact = export.artifact.as_object().expect("the artifact is an object");

        assert_eq!(artifact["refined"]["kind"], ARTIFACT_KIND);
        assert_eq!(artifact["refined"]["version"], ARTIFACT_VERSION);
        assert_eq!(artifact["target"]["file"], "audio_level_python_only.py");
        let hash = artifact["target"]["contentHash"]
            .as_str()
            .expect("contentHash is a string");
        assert!(hash.starts_with("sha256:"), "contentHash = {hash}");
        assert_eq!(hash.len(), "sha256:".len() + 64, "contentHash = {hash}");
        assert_eq!(&hash["sha256:".len()..], sha256_hex(&source).as_str());
        assert_eq!(artifact["runtime"]["band"], RUNTIME_BAND);
        // the fixture has no `__main__` block, so the harness key is
        // absent rather than guessed
        assert!(!artifact.contains_key("harness"));

        let functions = artifact["functions"]
            .as_object()
            .expect("functions is an object");
        // Every def either exports or is named in an omission — the
        // artifact never silently drops one.
        let exported: HashSet<&str> = functions.keys().map(|key| key.as_str()).collect();
        let omitted: HashSet<&str> = export
            .omissions
            .iter()
            .map(|omission| omission.function.as_str())
            .collect();
        for def in top_level_defs(&module) {
            let name = def.name.id.as_str();
            assert!(
                exported.contains(name) || omitted.contains(name),
                "'{name}' is neither exported nor named in an omission"
            );
        }

        for (name, entry) in functions {
            let rows = entry["entry"].as_array().expect("entry is an array");
            assert_eq!(rows.len(), 1, "'{name}' declares one parameter");
            let row = &rows[0];
            assert_eq!(row["name"], "samples");
            // `Annotated[list[Sample], Field(min_length=1)]` — a
            // sequence row, its element set present and its length floor
            // the declaration's own 1.
            let sequence = row
                .get("sequence")
                .unwrap_or_else(|| panic!("'{name}' states a sequence entry"));
            assert_eq!(sequence["lengthAtLeast"], 1);
            assert!(
                !sequence["element"]["forms"]
                    .as_array()
                    .expect("the element set carries forms")
                    .is_empty(),
                "'{name}' states an empty element set"
            );
            let returned = &entry["return"];
            assert!(
                !returned["set"]["forms"]
                    .as_array()
                    .expect("the return set carries forms")
                    .is_empty(),
                "'{name}' states an empty return set"
            );
            assert!(returned["stdoutPure"].is_boolean());
            assert!(
                entry["provenance"]["line"].as_i64().expect("line is a number") > 0,
                "'{name}' states a 1-based def line"
            );
            let said = entry["provenance"]["said"].as_str().expect("said is a string");
            assert!(said.contains("samples"), "said = {said:?}");
            assert!(said.contains("derive"), "said = {said:?}");
        }
    }
}
