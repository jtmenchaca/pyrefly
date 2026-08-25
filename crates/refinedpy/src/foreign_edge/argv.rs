use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprList;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::diagnostic_sentences;
use crate::env::Environment;

use super::keywords::literal_string;

/* ── recognition ──────────────────────────────────────────────────── */

/// A decline the recognizer already knows enough to name — distinct
/// from "not this shape at all," which owes no sentence.
pub(super) struct RecognitionDecline {
    pub(super) message: String,
    pub(super) range: TextRange,
}

/// The recognized runner words — argv[0] (plus, for a two-word runner,
/// argv[1]) that names the program the target `.ts` file runs under.
/// Every runner recognizes the REFERENCE (the argv genuinely names one
/// script) and, once the artifact declares the shared `es2023+` band,
/// discharges the runtime-identity premise identically — the band names
/// an ECMA-262 spec level, not one runtime binary (ruling, 2026-08-21).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Runner {
    Node,
    Deno,
    Bun,
    NpxTsx,
    /// argv holds exactly one element, a path-shaped literal with no
    /// runner word at all (`["./targets/cpp_level"]`) — the script IS
    /// argv[0]; there is no separate interpreter. This checker's own
    /// TypeScript-artifact reader has no producer for a compiled
    /// binary, so every recognized row of this shape reaches the
    /// artifact lookup and declines there, naming the compiled-binary
    /// construct rather than the generic no-fact sentence.
    CompiledBinary,
}

impl Runner {
    /// The exact runner text this call spells — carried into a decline
    /// sentence that names the runner (an unfit-input decline, an
    /// unrecognized script extension), never a category.
    pub(super) fn word(self) -> &'static str {
        match self {
            Runner::Node => "node",
            Runner::Deno => "deno",
            Runner::Bun => "bun",
            Runner::NpxTsx => "npx tsx",
            Runner::CompiledBinary => "the compiled binary",
        }
    }
}

/// One argv list read as `[<runner words>, <script>]` — the runner
/// identified and the script's own literal text resolved, independent
/// of which `subprocess.*` callee is being recognized (`run`,
/// `check_output`, and `Popen` all take the same argv shape).
pub(super) struct ArgvReading {
    pub(super) runner: Runner,
    pub(super) script_text: String,
}

/// Reads one `Expr::List` argv literal as `[runner_word(s), script]`:
/// exactly two elements for `Node`/`Bun` (`["node", script]`), exactly
/// three for `Deno`/`NpxTsx` (`["deno", "run", script]`,
/// `["npx", "tsx", script]`). Any other length, or a two/three-element
/// list whose runner word(s) do not match one of these four rows,
/// answers `None` — "some other program, nothing owed" for a
/// recognized-length list with an unrecognized word, and a decline
/// (owed by the caller, since the shape genuinely does not fit ANY
/// known runner) for every other length.
///
/// `None` is also the answer when argv[0] is not a written string
/// literal — an interpreter read through a variable is not a shape any
/// runner row here recognizes (`level_via_runner_variable`'s own row),
/// so the caller declines naming that specifically.
pub(super) fn argv_runner_and_script(
    argv_list: &ExprList,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ArgvReading, RecognitionDecline>> {
    let call_range = argv_list.range();
    match argv_list.elts.as_slice() {
        [interpreter, script] => {
            let Some(interpreter_text) = interpreter_text_of(interpreter, environment, kernel) else {
                return Some(Err(RecognitionDecline {
                    message: "this call's argv[0] is not a written string literal naming the interpreter"
                        .to_owned(),
                    range: call_range,
                }));
            };
            let runner = match interpreter_text.as_str() {
                "node" => Runner::Node,
                "bun" => Runner::Bun,
                _ => return None,
            };
            Some(script_text_of(script, environment, kernel).map(|script_text| ArgvReading { runner, script_text }))
        }
        [interpreter, second_word, script] => {
            let (Some(interpreter_text), Some(second_word_text)) =
                (literal_string(interpreter), literal_string(second_word))
            else {
                return Some(Err(RecognitionDecline {
                    message: "this call's argv[0] is not a written string literal naming the interpreter"
                        .to_owned(),
                    range: call_range,
                }));
            };
            let runner = match (interpreter_text, second_word_text) {
                ("deno", "run") => Runner::Deno,
                ("npx", "tsx") => Runner::NpxTsx,
                _ => return None,
            };
            Some(script_text_of(script, environment, kernel).map(|script_text| ArgvReading { runner, script_text }))
        }
        [only_element] => match compiled_binary_path_of(only_element, environment, kernel) {
            Some(script_text) => Some(Ok(ArgvReading { runner: Runner::CompiledBinary, script_text })),
            None => Some(Err(RecognitionDecline {
                message: "this call's argv does not hold exactly [\"node\", \"<script>.ts\"] (or a recognized \
                    deno/bun/npx-tsx runner row, or a bare compiled-binary path), so the checker cannot name the \
                    code that runs next"
                    .to_owned(),
                range: call_range,
            })),
        },
        _ => Some(Err(RecognitionDecline {
            message: "this call's argv does not hold exactly [\"node\", \"<script>.ts\"] (or a recognized \
                deno/bun/npx-tsx runner row), so the checker cannot name the code that runs next"
                .to_owned(),
            range: call_range,
        })),
    }
}

/// A single-element argv's own text, when it is PATH-shaped (`./`,
/// `../`, or `/`) — the compiled-binary row (`["./targets/cpp_level"]`):
/// argv[0] IS the script, since there is no runner word at all. Not
/// path-shaped (a bare word with no leading path marker, which is not a
/// recognized runner word either at this length) answers `None`, so the
/// caller's own "not this shape" decline still names the true absence
/// rather than misreading an arbitrary one-word argv as a binary path.
pub(super) fn compiled_binary_path_of(element: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<String> {
    let text = const_folded_text_of(element, environment, kernel)?;
    if text.starts_with("./") || text.starts_with("../") || text.starts_with('/') {
        Some(text)
    } else {
        None
    }
}

/// Reads the script element's own text: a written string literal
/// directly, a bare `Name` this body never rebinds that resolves
/// (through `environment.read`) to a known exact string — the
/// module-level-constant path (`TARGET_PATH = "./targets/level_ok.ts"`
/// used as `["node", TARGET_PATH]`) — or any OTHER expression this
/// checker's own string machinery folds to an exact value: an f-string
/// composed entirely of consts (`f"./targets/{name}.ts"` with `name` a
/// known exact string — `expressions::evaluate_fstring`'s own exact
/// tier), or a `+` concatenation of known exact strings
/// (`"./targets/" + name`, `expressions.rs`'s `Operator::Add` row on two
/// `exact_string_values`). `evaluate_expression` is the ONE reader for
/// both — this function never re-derives what that dispatcher already
/// computes; it only asks it and reads the answer back through the same
/// `exact_string_text` the module-constant path already uses.
///
/// Every expression `evaluate_expression` cannot fold to an exact string
/// (a parameter, a call this checker does not model exactly, an
/// `os.path.join(...)` — not modeled anywhere in this checker) declines
/// with the law-2 sentence: the path is computed, and the fix is to
/// spell it as a written string literal.
///
/// A script position always owes either a resolved reading or the
/// law-2 decline — never a bare `None` (that belongs to the caller's
/// own runner-word match, not to this function).
pub(super) fn script_text_of(
    script: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Result<String, RecognitionDecline> {
    match const_folded_text_of(script, environment, kernel) {
        Some(text) => Ok(text),
        None => Err(RecognitionDecline { message: diagnostic_sentences::script_path_not_a_literal(), range: script.range() }),
    }
}

/// Reads the interpreter element's (argv[0]'s) own text through the SAME
/// const-fold `script_text_of` already applies to the script position: a
/// written string literal directly, a bare `Name` this body never rebinds
/// that resolves to a known exact string (a local or module-level
/// constant — `level_via_runner_variable`'s own `runner = "node"` row),
/// or any other expression `evaluate_expression` folds to an exact
/// string. `None` is a genuine "not a runner row this reader can name" —
/// the SAME answer an unrecognized interpreter word already gives its
/// caller — never a decline of its own: naming which text this fold
/// found but did not recognize as a runner word is the caller's
/// (`argv_runner_and_script`'s) job, exactly as it already is for a
/// plain string literal spelling an unrecognized runner.
pub(super) fn interpreter_text_of(interpreter: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<String> {
    const_folded_text_of(interpreter, environment, kernel)
}

/// The shared const-fold `script_text_of`/`interpreter_text_of` both run:
/// a written string literal, a bare `Name` resolving (through
/// `environment.read`) to a known exact string, or any other expression
/// `evaluate_expression` folds to an exact string (an f-string composed
/// entirely of consts — `expressions::evaluate_fstring`'s own exact
/// tier — or a `+` concatenation of known exact strings,
/// `expressions.rs`'s `Operator::Add` row on two `exact_string_values`).
/// `evaluate_expression` is the ONE reader for both callers — this
/// function never re-derives what that dispatcher already computes; it
/// only asks it and reads the answer back through `exact_string_text`.
/// `None` when no tier folds the expression to an exact string (a
/// parameter, a call this checker does not model exactly, an
/// `os.path.join(...)` — not modeled anywhere in this checker).
pub(super) fn const_folded_text_of(expression: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<String> {
    if let Some(literal) = literal_string(expression) {
        return Some(literal.to_owned());
    }
    if let Expr::Name(name) = expression {
        if let Some(bound) = environment.read(name.id.as_str()) {
            if let Some(text) = exact_string_text(bound) {
                return Some(text);
            }
        }
    }
    let folded = crate::expressions::evaluate_expression(expression, environment, kernel);
    exact_string_text(&folded)
}

/// The exact text an `AbstractValue` carries, if it is a `Kind::Values`
/// state sorted `PrimitiveKind::String` — the same code-point-vector
/// shape every other file in this crate decodes locally
/// (`string_models.rs::exact_string_text`, reimplemented per file per
/// this crate's own no-shared-private-helper convention rather than
/// widening another module's visibility for one caller).
pub(super) fn exact_string_text(value: &AbstractValue) -> Option<String> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect())
}

/// Splits a shell command string on single spaces — the narrowest
/// tokenizer this row needs (no quoting). `None` for an empty command.
pub(super) fn tokenize_shell_command(command_text: &str) -> Option<Vec<&str>> {
    if command_text.is_empty() {
        return None;
    }
    Some(command_text.split(' ').collect())
}

/// Reads the leading `[runner(+word), script]` prefix off a tokenized
/// shell command, answering which `Runner` the leading tokens spelled
/// (the same tag `recognized_argv`'s `Expr::List` path already
/// resolves), the recognized "runner script" text, and whatever tokens
/// follow it. `None` when the leading tokens are not one of the four
/// recognized runner rows at all.
pub(super) fn split_runner_and_script_tagged<'a>(tokens: &'a [&'a str]) -> Option<(Runner, String, &'a [&'a str])> {
    match tokens {
        [runner_word, script, rest @ ..] if is_recognized_runner_word(runner_word) => {
            let runner = if *runner_word == "node" { Runner::Node } else { Runner::Bun };
            Some((runner, format!("{runner_word} {script}"), rest))
        }
        [runner_word, second_word, script, rest @ ..] if is_recognized_two_word_runner(runner_word, second_word) => {
            let runner = if *runner_word == "deno" { Runner::Deno } else { Runner::NpxTsx };
            Some((runner, format!("{runner_word} {second_word} {script}"), rest))
        }
        _ => None,
    }
}

/// Whether a token is a recognized one-word runner (`node`, `bun`).
pub(super) fn is_recognized_runner_word(word: &str) -> bool {
    word == "node" || word == "bun"
}

/// Whether two tokens are a recognized two-word runner (`deno run`,
/// `npx tsx`).
pub(super) fn is_recognized_two_word_runner(first: &str, second: &str) -> bool {
    (first == "deno" && second == "run") || (first == "npx" && second == "tsx")
}

/// The argv list and its resolved runner/script — read once, shared by
/// every callee's own recognition. `None` propagates a not-this-shape
/// answer (an unrecognized argv[0] at a recognized length: "some other
/// program, nothing owed"); `Some(Err(...))` is a decline the caller
/// returns unchanged.
pub(super) fn recognized_argv(
    call: &ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ArgvReading, RecognitionDecline>> {
    let call_range = call.range();
    let [argv] = call.arguments.args.as_ref() else {
        return Some(Err(RecognitionDecline {
            message: "this call passes other than one positional argv argument, and the checker models \
                only a written argv list naming one script"
                .to_owned(),
            range: call_range,
        }));
    };
    let Expr::List(argv_list) = argv else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv is not one written list literal, so the checker cannot name the \
                code that runs next — no edge is modeled here"
                .to_owned(),
            range: call_range,
        }));
    };
    argv_runner_and_script(argv_list, environment, kernel)
}

/// `asyncio.create_subprocess_exec`'s own argv reading: the runner and
/// script ride the call's VARIADIC positional arguments (`program, *args`)
/// rather than one list literal — `["node", script]`/`["deno", "run",
/// script]`/`["npx", "tsx", script]` reread as `call.arguments.args`
/// holding exactly two or three positional elements, the same runner-word
/// match and script resolution `argv_runner_and_script` already applies
/// to a list literal's own elements.
pub(super) fn asyncio_argv_runner_and_script(
    call: &ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ArgvReading, RecognitionDecline>> {
    let call_range = call.range();
    match call.arguments.args.as_ref() {
        [interpreter, script] => {
            let Some(interpreter_text) = literal_string(interpreter) else {
                return Some(Err(RecognitionDecline {
                    message: "this call's leading positional argument is not a written string literal \
                        naming the interpreter"
                        .to_owned(),
                    range: call_range,
                }));
            };
            let runner = match interpreter_text {
                "node" => Runner::Node,
                "bun" => Runner::Bun,
                _ => return None,
            };
            Some(script_text_of(script, environment, kernel).map(|script_text| ArgvReading { runner, script_text }))
        }
        [interpreter, second_word, script] => {
            let (Some(interpreter_text), Some(second_word_text)) =
                (literal_string(interpreter), literal_string(second_word))
            else {
                return Some(Err(RecognitionDecline {
                    message: "this call's leading positional argument is not a written string literal \
                        naming the interpreter"
                        .to_owned(),
                    range: call_range,
                }));
            };
            let runner = match (interpreter_text, second_word_text) {
                ("deno", "run") => Runner::Deno,
                ("npx", "tsx") => Runner::NpxTsx,
                _ => return None,
            };
            Some(script_text_of(script, environment, kernel).map(|script_text| ArgvReading { runner, script_text }))
        }
        _ => Some(Err(RecognitionDecline {
            message: "this call's positional arguments do not hold exactly (\"node\", \"<script>.ts\") (or \
                a recognized deno/bun/npx-tsx runner row), so the checker cannot name the code that runs \
                next"
                .to_owned(),
            range: call_range,
        })),
    }
}

/// Whether the script text names a `.ts` file — the one extension this
/// edge models a fact for. A compiled-binary row names no TypeScript
/// source at all (the argv's own text names the compiled binary
/// itself), so this extension premise does not apply to it — its own
/// artifact-lookup decline (`compiled_binary_no_fact_sentence`) is the
/// construct this shape blocks on, never a wrong-extension sentence.
pub(super) fn script_extension_decline(script_text: &str, runner: Runner, call_range: TextRange) -> Option<RecognitionDecline> {
    if runner == Runner::CompiledBinary || script_text.ends_with(".ts") {
        return None;
    }
    Some(RecognitionDecline {
        message: format!(
            "this call runs {} on {script_text}, which is not a .ts file — the checker models the edge \
            only where the argv names TypeScript source it can read a fact for",
            runner.word()
        ),
        range: call_range,
    })
}

/// The `.ts` path resolved against the checked file's own directory —
/// mirroring the Go twin's own `foreignEdgeOf`: a relative argv entry
/// is relative to the file that wrote it, never the eventual run's cwd.
///
/// This module has no source-file handle of its own (unlike the Go
/// walk, which reads `ast.GetSourceFileOfNode`), so callers resolve a
/// RELATIVE script name against the checked file's directory at the
/// artifact-read call site; here the raw argv text is kept as-is and
/// resolution happens where the checked file's own path is known
/// (`check.rs`'s entry point already threads that path for module
/// resolution). An absolute script name is returned unchanged.
pub(super) fn resolve_target_path(script_text: &str) -> String {
    script_text.to_owned()
}

