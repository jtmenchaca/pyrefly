//! The cross-language call edge, recognized in the walk — the REVERSE
//! pair of refined-ts-go's `walk/foreign_edge.go` (TS calling Python):
//! here Python calls out to a TypeScript body over stdin/stdout JSON,
//! reads back the target's own kernel-derived return fact, and attaches
//! it to the `json.loads(...)` node that reads the captured stdout.
//!
//! Recognized shape (`docs/one-checker/reverse-pair.md`, Half B):
//!
//! ```python
//! result = subprocess.run(
//!     ["node", "./audio_level.ts"],
//!     input=json.dumps(boosted),
//!     capture_output=True,
//!     text=True,
//! )
//! return json.loads(result.stdout)
//! ```
//!
//! Two other `subprocess` callees recognize the same argv/payload shape:
//! `subprocess.check_output(...)` (the captured text is the CALL's own
//! return, read bare — `json.loads(result)`, never `result.stdout`) and
//! the two-statement `subprocess.Popen(...)` / `<stdout>, _ = <name>
//! .communicate(json.dumps(...))` pair, where `.communicate()`'s own
//! call carries the payload the `Popen(...)` call itself does not.
//!
//! The runner word at argv[0] (plus, for a two-word runner, argv[1])
//! also recognizes beyond plain `"node"`: `"deno" "run"`, `"bun"`, and
//! `"npx" "tsx"` all name a real script the same way `"node"` does. The
//! band this checker's TypeScript pins commit to (`es2023+`) names an
//! ECMA-262 spec level, not one runtime binary (ruling, 2026-08-21), so
//! every recognized runner discharges the runtime-identity premise
//! identically once the artifact declares that band — the artifact
//! reader's own band check (`foreign_edge_artifact.rs`) is the only gate,
//! and it applies the same way regardless of which runner the call names.
//!
//! argv[1] (the script) also resolves through a module-level constant
//! this body reads (`TARGET_PATH = "./x.ts"` used as `["node",
//! TARGET_PATH]`) — any other non-literal shape (an f-string, a
//! parameter) declines with the law-2 sentence naming the fixable
//! written-literal respelling.
//!
//! A SIBLING carrier: `subprocess.run(["node", "<script>.ts",
//! json.dumps(<payload>)], capture_output=True, text=True)` — the
//! payload rides the third argv element (`process.argv[2]`, node's own
//! convention) rather than stdin, and carries no `input=` keyword at
//! all (its presence alongside an argv payload is a real double-channel
//! ambiguity, declined rather than silently picking one). The target's
//! own artifact must declare a matching `surface.kind == "argv-json"`
//! (with the same `argIndex`) for this shape to apply; an argv payload
//! against a `stdin-json` target (or the reverse) declines naming the
//! channel mismatch — recognized shapes on both sides, transports that
//! do not meet.
//!
//! A THIRD carrier — a named TEMP FILE — sends the payload through
//! neither a pipe nor an argv element's own text: `with tempfile
//! .NamedTemporaryFile(mode="w", suffix=".json", delete=False) as
//! handle: json.dump(<payload>, handle); temp_path = handle.name`
//! immediately followed by `subprocess.run(["node", "<script>.ts",
//! temp_path], capture_output=True, text=True)`. The argv element
//! carries the file's PATH (a bare name, never `json.dumps(...)`), and
//! the target reads its JSON from that file (node's own
//! `readFileSync(process.argv[2], "utf8")`). This is a THREE-STATEMENT
//! unit (`recognize_temp_file_edge`): the `with`-block itself supplies
//! the payload and the bound path name, and the call one statement
//! later must name that SAME bound name at argv[2] — a reassignment of
//! `temp_path` between the dump and the call leaves the checker unable
//! to prove the file the call reads is the file `json.dump` wrote, so
//! it stays undetermined naming the rebind. The target's own artifact
//! must declare a matching `surface.kind == "file-json"` (with the same
//! `argIndex`) for this shape to apply; a temp-file payload against a
//! `stdin-json` or `argv-json` target (or the reverse) declines naming
//! the channel mismatch, the same way the argv-json sibling does.
//!
//! CROSS-LANGUAGE-EDGE.md §2's corollary makes this a real edge and not
//! a manifest: the argv deterministically NAMES the code that runs
//! next, so the checker treats the call the way it treats an import.
//! §11 is this exact spelling; §4 is the JSON transport model both legs
//! apply; §5 is the list of premises the crossing rests on.
//!
//! WHAT THE ROUTE DOES, in order (mirrors the Go twin's own banner):
//!
//!  1. RECOGNIZE the call: an `Assign` of one name from a recognized
//!     `subprocess` callee, with a written argv list naming a runner and
//!     a script, and every required keyword. Anything unrecognized
//!     declines, and every decline NAMES what broke.
//!  2. READ the target's exported fact off disk through the sibling's
//!     `read_foreign_ts_artifact` — target integrity, runtime identity,
//!     and harness shape are the artifact reader's own premises.
//!  3. DISCHARGE the outbound leg's premises against the value actually
//!     being stringified: NaN-freedom (NaN stringifies to `null`, so the
//!     target never sees the number the caller sent) and the crossing
//!     fit (the argument's element set inside the entry's, its length
//!     floor at or above the entry's). A fit FAILURE is not a decline:
//!     it is a 7001 at the call, because the value can escape what the
//!     target states it admits.
//!  4. DISCHARGE channel purity and ATTACH the return fact to the
//!     `json.loads(result.stdout)` node — the sole consumer of the
//!     captured stdout, found the same way the Go twin's
//!     `soleParseConsumerOf` finds its `JSON.parse` node.
//!
//! The attach rides `Environment::set_evaluated_node`, the seam the
//! relational-sum lane already uses for a value no re-walk can reach —
//! `check.rs`'s own return-position quotient publish (check.rs:1975-1978)
//! is the exact precedent this route follows.
//!
//! TRUST GRADE. The attached fact is stamped `TrustSpec`, not
//! `TrustProved` — the mirror of the Go twin's own reasoning
//! (`foreignReturnValue`'s doc): every premise here is a real check, but
//! the crossing itself rests on cited spec behaviour (the JSON number
//! round-trip) this tree has not proved as a kernel theorem.
//!
//! `json.loads` always answers a Python `float` for a JSON number
//! whose text carries a fractional/exponent part (library/json.rst's
//! conversion table — "number (int)" only when the JSON text itself
//! has no such part AND the loader's own `parse_int` is not
//! overridden, which this checker does not read). The CHECKER's own
//! sort tag on the crossed value does not stamp Float uniformly over
//! this ambiguity; it reads the target's declared return set for its
//! own `Integer` form the same way a declared position's sort is read
//! (`foreign_return_value`'s doc) — an all-integer return reads
//! Integer, and only an unmarked or genuinely fractional return reads
//! Float.
//!
//! CORNER CHECK: the return set's own corner values must be ones the
//! TypeScript target's own serializer actually carries. The mechanism is
//! NOT that legal JSON text (RFC 8259) has no token for ±Infinity — a
//! JSON leg can carry it fine (`1e999` is legal JSON text and parses to
//! Infinity in both runtimes). The mechanism is `JSON.stringify` itself:
//! it serializes a non-finite Number as the bare literal `null`
//! (ECMA-262's `SerializeJSONProperty`, the finiteness check on a Number
//! value), a value outside the claimed numeric set landing at this leg's
//! own `json.loads` consumer. A return set whose corners admit ±Infinity
//! degrades to a named undetermined instead of binding the set as stated
//! (`foreign_return_value_or_undetermined`); NaN is already excluded
//! from every `RefinedSet` at construction (the boundary ruling), so
//! only the two infinite corners need the check.


use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::Stmt;

#[cfg(test)]
use crate::foreign_edge_artifact::ForeignTsFunctionFact;
use crate::env::Environment;

mod argv;
mod cases;
mod crossing;
mod discharge;
mod edge_types;
mod keywords;
mod parse_consumer;
mod recognize;

#[cfg(test)]
mod tests;

pub(crate) use keywords::literal_true;
pub use edge_types::ForeignEdge;
pub use edge_types::ForeignEdgeOutcome;
use edge_types::Channel;
use edge_types::ResultRead;

use discharge::finish_recognized_edge;
use discharge::finish_recognized_edge_from_start;
use recognize::recognize_foreign_edge;
use recognize::recognize_popen_context_manager_edge;
use recognize::recognize_subprocess_callee;
use recognize::recognize_temp_file_edge;


/// Recognizes a cross-language call at `statements[index]` and, on all
/// premises green, answers the override the caller publishes for the
/// rest of the body's walk.
///
/// Answers `None` for every statement that is not this shape — the
/// ordinary walk is untouched and pays one recognizer's worth of
/// syntax tests. A recognized edge that cannot be completed answers an
/// outcome carrying a decline sentence: an edge the checker sees and
/// cannot serve is a work-queue item, never a silence.
pub fn foreign_edge_at(
    statements: &[Stmt],
    index: usize,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Option<ForeignEdgeOutcome> {
    // The idiomatic `with subprocess.Popen([...]) as process:` wrapping
    // (`level_via_popen_context_manager`'s own shape) puts its OWN
    // consumer — the `.communicate()` assign and, later, the
    // `json.loads(...)` read — inside the WITH-BLOCK's own body, never
    // as a sibling of the `with` statement in `statements`. Every other
    // recognized shape (a plain `Assign`, or the temp-file `with`) keeps
    // scanning `statements` exactly as before; only this one shape scans
    // its own nested body instead.
    if let Stmt::With(with_stmt) = &statements[index] {
        if recognize_temp_file_edge(with_stmt, statements, index, environment, kernel).is_none() {
            if let Some(edge) = recognize_popen_context_manager_edge(with_stmt, environment, kernel) {
                return finish_recognized_edge(edge, &with_stmt.body, environment, kernel, entry_directory);
            }
        }
    }
    let edge = recognize_foreign_edge(statements, index, environment, kernel)?;
    finish_recognized_edge(edge, statements, environment, kernel, entry_directory)
}


/// Recognizes a cross-language call directly off a walrus-bound `<name>
/// := subprocess.<callee>(...)` inside an `if`/`elif` TEST (`level_via_
/// walrus_result`'s own shape: `Stmt::If`, never an `Assign`/`With`, so
/// `foreign_edge_at`'s own `statements[index]` dispatch structurally
/// never reaches it) and, on all premises green, answers the override
/// the caller publishes for the rest of the ARM body's walk.
///
/// `target`/`call` are the walrus's own `Expr::Named::target`/`value`
/// (already destructured by the caller, which knows the walrus shape);
/// `arm_body` is the taken arm's own statement list — the return leg's
/// sole-consumer scan runs over THAT list (`sole_parse_consumer_of`
/// reads forward from `arm_scan_from`), since the `json.loads(...)`
/// consumer sits inside the arm, never as a sibling of the outer `if`.
/// Answers `None` for every callee this crate does not recognize at all
/// — the same "not this shape, no sentence owed" contract
/// `recognize_foreign_edge` keeps for its own Assign path.
pub fn foreign_edge_at_walrus_call(
    call: &ExprCall,
    target: &ExprName,
    arm_body: &[Stmt],
    arm_scan_from: usize,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Option<ForeignEdgeOutcome> {
    let edge = recognize_subprocess_callee(call, target, arm_body, arm_scan_from, environment, kernel)?;
    // The walrus-bound call sits inside the `if` TEST, never as a member
    // of `arm_body` — there is no call STATEMENT for the return leg's
    // scan to skip past, unlike the `Stmt::Assign`/`Stmt::With` shapes
    // `finish_recognized_edge`'s other callers supply. The whole arm
    // body is offered to the consumer scan from its own start.
    finish_recognized_edge_from_start(edge, arm_body, environment, kernel, entry_directory)
}


