use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_text_size::Ranged;

use crate::diagnostic_sentences;
use crate::env::Environment;

use super::super::argv::RecognitionDecline;
use super::super::argv::Runner;
use super::super::argv::recognized_argv;
use super::super::argv::resolve_target_path;
use super::super::argv::script_extension_decline;
use super::super::argv::script_text_of;
use super::super::keywords::json_dumps_argument_of;
use super::super::keywords::literal_string;
use super::super::keywords::subprocess_run_argv_json_keywords_of;
use super::super::keywords::subprocess_run_keywords_of;
use super::super::Channel;
use super::super::ForeignEdge;
use super::super::ResultRead;

/// `<name> = subprocess.run(["node", "<script>.ts"], input=json.dumps(
/// <payload>), capture_output=True, text=True)` — the result reads back
/// at `<name>.stdout`. The sibling argv-json shape (`["node",
/// "<script>.ts", json.dumps(<payload>)]`, no `input=` keyword) is tried
/// first: it is a real ambiguity with the ordinary two-element-argv
/// shape only when BOTH an argv payload and `input=` are present, which
/// `argv_json_call_of` itself declines naming the double channel.
pub(super) fn recognize_subprocess_run(
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
    index: usize,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    if let Some(argv_json) = argv_json_call_of(call, target, environment, index, kernel) {
        return Some(argv_json);
    }
    let call_range = call.range();
    let reading = match recognized_argv(call, environment, kernel)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    let (payload, keywords_decline) = subprocess_run_keywords_of(call);
    if let Some(decline) = keywords_decline {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(payload) = payload else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call runs {} on {} and sends it no json.dumps(...) input, so nothing crosses out on \
                stdin and the transport model has no outbound leg to apply",
                reading.runner.word(),
                reading.script_text
            ),
            range: call_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&reading.script_text),
        payload,
        channel: Channel::Stdin,
        result_name: target.id.as_str().to_owned(),
        result_read: ResultRead::StdoutAttribute,
        consumer_scan_from: index,
        runner: reading.runner,
    }))
}

/// `<name> = subprocess.run(["node", "<script>.ts", json.dumps(<payload>)],
/// capture_output=True, text=True)` — the payload rides the third argv
/// element (node's own convention: `process.argv[2]`) rather than
/// stdin. `None` when this is not that three-element shape at all (the
/// ordinary two-element stdin call, an unrelated argv arity, or a
/// three-element deno/npx-tsx runner row whose own trailing element is
/// the SCRIPT, not a `json.dumps(...)` payload — `literal_string` on
/// that element fails `json_dumps_argument_of`'s own call-shape check,
/// so it falls through here unchanged). `Some(Err(...))` when the shape
/// reads as an argv payload but something about it stops recognition:
/// an unreadable runner/script, a wrong extension, or `input=` ALSO
/// present (the double-channel decline — two crossing values are named
/// and this checker recognizes exactly one transport per call).
pub(super) fn argv_json_call_of(
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
    index: usize,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let call_range = call.range();
    let [argv] = call.arguments.args.as_ref() else {
        return None;
    };
    let Expr::List(argv_list) = argv else {
        return None;
    };
    let [interpreter, script, third] = argv_list.elts.as_slice() else {
        return None;
    };
    let payload = json_dumps_argument_of(third)?;
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
                    "this call's argv names {interpreter_text} as the third-position payload's runner, and \
                    the argv-json shape recognizes only node/bun at that position"
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
        channel: Channel::Argv { arg_index: 2 },
        result_name: target.id.as_str().to_owned(),
        result_read: ResultRead::StdoutAttribute,
        consumer_scan_from: index,
        runner,
    }))
}
