//! `subprocess.run(...)`'s own `.stdout` sort — a small model that
//! lives beside the datetime family for historical reasons (the file
//! this module was split from), not because it models a temporal
//! construct.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustSpec;
use refined_sets::codepoint_sets::strings;
use ruff_python_ast::Expr;

use crate::env::Environment;
use crate::foreign_edge;

/// `subprocess.run([...], ..., capture_output=True, text=True)` —
/// library/subprocess.rst, `class:: CompletedProcess`: "args, returncode,
/// stdout, stderr" are the instance's own attributes, and `run`'s own
/// entry states `capture_output=True` sets `stdout`/`stderr` to
/// `PIPE`, while `text=True` (an alias for `universal_newlines`) makes
/// every captured stream "opened in text mode" — a `str`, never `bytes`.
/// Modeled ONLY as far as `.stdout`'s own SORT: an OBJECT (`Kind::Object`,
/// untagged `source` — the same untagged shape `cross_module.rs`'s own
/// module object carries, so `evaluate_attribute_read`'s tail falls
/// straight to the plain `instances::field_read` linear scan) with one
/// `ObjectKey` named `stdout`, holding the whole-strings ground
/// (`codepoint_sets::strings()`, `C*`) — the same untagged String-sorted
/// `Kind::Set` `__name__` reads (this file's own `Expr::Name` arm). No
/// OTHER `CompletedProcess` field (`returncode`, `stderr`, `args`) is
/// modeled: this row exists to give `.stdout` a SORT for a body that
/// reads it some other way than `json.loads(...)` (`foreign_edge.rs`'s
/// own `json.loads(result.stdout)` consumer path owns that shape
/// separately, and runs BEFORE this construction ever matters — a
/// recognized foreign edge overrides its own consumer node directly;
/// this row only affects `result` itself and every OTHER read of it,
/// `d-data-legs.py`'s own `level_via_raw_stdout`: `float(result.stdout)`,
/// never parsed as JSON).
///
/// Declines (`None`) unless the module name is `subprocess` (not locally
/// shadowed — the same check every other `subprocess`/module recognizer
/// in this crate applies), the attribute called is `run`, and BOTH
/// `capture_output=True` and `text=True` appear among the call's
/// keywords: away from that exact pair, `.stdout` is not provably a
/// `str` at all (no `capture_output=True` leaves stdout un-captured
/// entirely; no `text=True` leaves it `bytes`), so the whole construction
/// declines rather than guess the sort.
pub(in crate::expressions) fn subprocess_run_construction_value(attribute: &ruff_python_ast::ExprAttribute, call: &ruff_python_ast::ExprCall, environment: &Environment) -> Option<AbstractValue> {
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "subprocess" || environment.read("subprocess").is_some() {
        return None;
    }
    if attribute.attr.as_str() != "run" {
        return None;
    }
    let mut capture_output_true = false;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return None;
        };
        match name.as_str() {
            "capture_output" => capture_output_true = foreign_edge::literal_true(&keyword.value),
            "text" => text_true = foreign_edge::literal_true(&keyword.value),
            _ => {}
        }
    }
    if !capture_output_true || !text_true {
        return None;
    }
    let keys = vec![ObjectKey { name: "stdout".to_owned(), numeric: false, value: known_set(strings(), None, TrustSpec, SetKindTag::None) }];
    Some(known_object(keys, None, true, TrustSpec, false))
}
