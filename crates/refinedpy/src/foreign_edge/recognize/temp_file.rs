use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtWith;
use ruff_text_size::Ranged;

use crate::diagnostic_sentences;
use crate::env::Environment;

use super::super::argv::RecognitionDecline;
use super::super::argv::Runner;
use super::super::argv::resolve_target_path;
use super::super::argv::script_extension_decline;
use super::super::argv::script_text_of;
use super::super::keywords::literal_string;
use super::super::keywords::subprocess_run_argv_json_keywords_of;
use super::super::keywords::temp_file_keywords_of;
use super::super::parse_consumer::statement_writes_name;
use super::super::Channel;
use super::super::ForeignEdge;
use super::super::ResultRead;
use super::as_bare_name;
use super::popen::is_subprocess_run_call;

/// `with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
/// delete=False) as handle: json.dump(<payload>, handle); temp_path =
/// handle.name` immediately followed by `<name> = subprocess.run(
/// ["node", "<script>.ts", temp_path], capture_output=True, text=True)`
/// — the payload crosses through a NAMED TEMP FILE rather than stdin or
/// a `json.dumps(...)` argv element: the `with`-block's own two
/// statements write the payload to the file and bind its path to
/// `temp_path`, and the call's third argv element is that SAME bound
/// name, read bare (never `json.dumps(...)`, since the file already
/// carries the JSON text).
///
/// This is a THREE-STATEMENT unit, one further than `Popen`'s own
/// two-statement lookahead: the `with` statement (recognized here, at
/// `index`) supplies the payload and the path name, and the call the
/// checker still must find sits at `index + 1` — the same
/// one-statement-further lookahead `Popen`'s own recognition applies
/// for `.communicate()`, widened by the one extra statement the
/// `with`-block's own body contributes before the call is even reached.
///
/// CARRIER PREMISE: the bytes `json.dump` writes to the file are the
/// bytes the target reads back at that same path — no intervening
/// write to the file and no reassignment of `temp_path` between the
/// dump and the call. The JSON transport model this shares with
/// `stdin-json`/`argv-json` rests on the identical round-trip premise;
/// only the carrier (a file, rather than a pipe or an argv element)
/// differs.
///
/// `None` — not this shape at all (the `with`'s own context expression
/// is not `tempfile.NamedTemporaryFile(...)`, or a shadowed `tempfile`
/// name). `Some(Err(...))` — the `with`-block is recognizably this
/// shape and something about its spelling, or the call that must
/// follow it, stops the resolution.
pub(in crate::foreign_edge) fn recognize_temp_file_edge(
    with_stmt: &StmtWith,
    statements: &[Stmt],
    index: usize,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let with_range = with_stmt.range();
    let [item] = with_stmt.items.as_slice() else {
        return None;
    };
    if !is_named_temporary_file_call(&item.context_expr, environment) {
        return None;
    }
    let Some(handle_name) = item.optional_vars.as_deref().and_then(as_bare_name) else {
        return Some(Err(RecognitionDecline {
            message: "this tempfile.NamedTemporaryFile(...) with-statement binds no bare name via 'as', so \
                the checker cannot find the handle json.dump(...) writes through"
                .to_owned(),
            range: with_range,
        }));
    };
    let temp_file_keywords_decline = temp_file_keywords_of(&item.context_expr);
    if let Some(decline) = temp_file_keywords_decline {
        return Some(Err(RecognitionDecline { message: decline, range: with_range }));
    }
    let [dump_statement, path_statement] = with_stmt.body.as_slice() else {
        return Some(Err(RecognitionDecline {
            message: "this tempfile.NamedTemporaryFile(...) with-block does not hold exactly \
                `json.dump(<payload>, <handle>)` followed by `<name> = <handle>.name`, so the checker \
                cannot find the payload this call writes to the temp file"
                .to_owned(),
            range: with_range,
        }));
    };
    let Some(payload) = json_dump_payload_of(dump_statement, handle_name) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this with-block's first statement is not exactly json.dump(<payload>, {handle_name}), so \
                the checker cannot find the value written to the temp file"
            ),
            range: with_range,
        }));
    };
    let Some(temp_path_name) = handle_name_binding_of(path_statement, handle_name) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this with-block's second statement is not exactly `<name> = {handle_name}.name`, so the \
                checker cannot find the bound path the call must name"
            ),
            range: with_range,
        }));
    };
    // Scans forward for the subprocess.run(...) call that reads the temp
    // file back — the SAME discipline the return leg's own
    // `sole_parse_consumer_of` applies to its result name: an unrelated
    // statement in between is not itself a blocker
    // (`level_via_call_then_unrelated_then_parse`'s own precedent), but a
    // WRITE to `temp_path_name` before the call is found means the file
    // the call would name is no longer provably the file `json.dump`
    // wrote, so that is named and declined rather than silently
    // skipped past.
    let mut call_statement: Option<(usize, &StmtAssign, &ExprCall)> = None;
    for (offset, statement) in statements[index + 1..].iter().enumerate() {
        if let Stmt::Assign(assign) = statement {
            if let Expr::Call(call) = assign.value.as_ref() {
                if is_subprocess_run_call(call, environment) {
                    call_statement = Some((index + 1 + offset, assign, call));
                    break;
                }
            }
        }
        if statement_writes_name(statement, &temp_path_name) {
            return Some(Err(RecognitionDecline {
                message: format!(
                    "{temp_path_name} is written again before a subprocess.run(...) call reads it, so the \
                    file that call would name is not provably the file json.dump wrote"
                ),
                range: statement.range(),
            }));
        }
    }
    let Some((call_position, assign, call)) = call_statement else {
        return Some(Err(RecognitionDecline {
            message: "this temp-file with-block writes the payload out, and no subprocess.run(...) call \
                follows it in this body to read the file back"
                .to_owned(),
            range: with_range,
        }));
    };
    let [Expr::Name(target)] = assign.targets.as_slice() else {
        return Some(Err(RecognitionDecline {
            message: "the subprocess.run(...) call reading this temp file does not bind one plain name, so \
                the checker cannot find the call's own captured result"
                .to_owned(),
            range: with_range,
        }));
    };
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
    let [interpreter, script, third] = argv_list.elts.as_slice() else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv does not hold exactly [\"node\", \"<script>.ts\", <temp path name>], \
                so the checker cannot name the code that runs next"
                .to_owned(),
            range: call_range,
        }));
    };
    // the third element must be a BARE NAME reading the SAME binding the
    // with-block produced — a json.dumps(...) call there is the sibling
    // argv-json shape, not this one, and any other shape does not name
    // the temp file's own path at all
    let Some(third_name) = as_bare_name(third) else {
        return Some(Err(RecognitionDecline {
            message: "this call's third argv element is not the bare name the with-block bound to the \
                temp file's path, so the checker cannot tell that this call reads that file back"
                .to_owned(),
            range: call_range,
        }));
    };
    if third_name != temp_path_name {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call's third argv element names {third_name}, and the with-block bound the temp \
                file's path to {temp_path_name} — the checker cannot tell these name the same file"
            ),
            range: call_range,
        }));
    }
    let Some(interpreter_text) = literal_string(interpreter) else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv[0] is not a written string literal naming the interpreter".to_owned(),
            range: call_range,
        }));
    };
    let runner = match interpreter_text {
        "node" => Runner::Node,
        "bun" => Runner::Bun,
        _ => {
            return Some(Err(RecognitionDecline {
                message: format!(
                    "this call's argv names {interpreter_text} as the temp-file shape's runner, and this \
                    checker recognizes only node/bun at that position"
                ),
                range: call_range,
            }));
        }
    };
    let script_text = match script_text_of(script, environment, kernel) {
        Ok(text) => text,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&script_text, runner, call_range) {
        return Some(Err(decline));
    }
    let (input_present, keywords_decline) = subprocess_run_argv_json_keywords_of(call);
    if let Some(decline) = keywords_decline {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    if input_present {
        return Some(Err(RecognitionDecline {
            message: diagnostic_sentences::foreign_edge_double_channel_declared(),
            range: call_range,
        }));
    }
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&script_text),
        payload,
        channel: Channel::File { arg_index: 2 },
        result_name: target.id.as_str().to_owned(),
        result_read: ResultRead::StdoutAttribute,
        consumer_scan_from: call_position,
        runner,
    }))
}

/// Whether an expression is exactly `tempfile.NamedTemporaryFile(...)` —
/// a shadowed `tempfile` name is not the module, mirroring every other
/// recognizer's shadow-on-rebind check.
pub(super) fn is_named_temporary_file_call(expression: &Expr, environment: &Environment) -> bool {
    let Expr::Call(call) = expression else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "tempfile"
        && environment.read("tempfile").is_none()
        && attribute.attr.as_str() == "NamedTemporaryFile"
}

/// Reads `json.dump(<payload>, <handle_name>)` — exactly two positional
/// arguments, no keywords, the second a bare name matching `handle_name`.
/// Answers the payload expression.
pub(super) fn json_dump_payload_of(statement: &Stmt, handle_name: &str) -> Option<Expr> {
    let Stmt::Expr(expr_stmt) = statement else {
        return None;
    };
    let Expr::Call(call) = expr_stmt.value.as_ref() else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "json" || attribute.attr.as_str() != "dump" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [payload, handle_arg] = call.arguments.args.as_ref() else {
        return None;
    };
    if as_bare_name(handle_arg) != Some(handle_name) {
        return None;
    }
    Some(payload.clone())
}

/// Reads `<name> = <handle_name>.name` — the with-block's second
/// statement, binding the temp file's own path to a plain name. Answers
/// the bound name's own text.
pub(super) fn handle_name_binding_of(statement: &Stmt, handle_name: &str) -> Option<String> {
    let Stmt::Assign(assign) = statement else {
        return None;
    };
    let [Expr::Name(target)] = assign.targets.as_slice() else {
        return None;
    };
    let Expr::Attribute(attribute) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return None;
    };
    if receiver.id.as_str() != handle_name || attribute.attr.as_str() != "name" {
        return None;
    }
    Some(target.id.as_str().to_owned())
}
