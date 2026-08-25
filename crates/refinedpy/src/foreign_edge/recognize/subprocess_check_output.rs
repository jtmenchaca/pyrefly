use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_text_size::Ranged;

use crate::env::Environment;

use super::super::argv::RecognitionDecline;
use super::super::argv::recognized_argv;
use super::super::argv::resolve_target_path;
use super::super::argv::script_extension_decline;
use super::super::keywords::subprocess_check_output_keywords_of;
use super::super::Channel;
use super::super::ForeignEdge;
use super::super::ResultRead;

/// `<name> = subprocess.check_output(["node", "<script>.ts"], input=
/// json.dumps(<payload>), text=True)` — the result IS the captured
/// stdout text directly (`library/subprocess.rst`: "the return value is
/// the command's output"), so the sole consumer reads `<name>` bare,
/// never `<name>.stdout`. No `capture_output` keyword exists for this
/// callee (`check_output` always captures), so it is not read here.
pub(super) fn recognize_subprocess_check_output(
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
    index: usize,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let call_range = call.range();
    let reading = match recognized_argv(call, environment, kernel)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    let (payload, keywords_decline) = subprocess_check_output_keywords_of(call);
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
        result_read: ResultRead::Bare,
        consumer_scan_from: index,
        runner: reading.runner,
    }))
}
