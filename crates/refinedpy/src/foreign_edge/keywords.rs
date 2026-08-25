use ruff_python_ast::ConversionFlag;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::InterpolatedStringElement;

/// Reads the `asyncio.create_subprocess_exec` keyword arguments:
/// `stdin=asyncio.subprocess.PIPE` and `stdout=asyncio.subprocess.PIPE` —
/// BOTH required (`.communicate()`'s own call, not this one, carries the
/// payload); any other keyword declines. No `text=True` keyword exists
/// for this callee at all (`library/asyncio-subprocess.rst`: the stream
/// always carries bytes), so it is neither read nor required here, unlike
/// `subprocess_popen_keywords_of`'s own sync check. An explicitly
/// non-PIPE `stdout` (a real file handle, `asyncio.subprocess.DEVNULL`,
/// anything else) refuses recognition with the same "cannot read the
/// target's stdout back" sentence the sync Popen row already speaks —
/// one channel-refusal sentence family, not a second one for the async
/// spelling.
pub(super) fn asyncio_create_subprocess_exec_keywords_of(call: &ExprCall) -> Option<String> {
    let mut stdin_pipe = false;
    let mut stdout_pipe = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return Some(
                "this call passes a keyword argument through **, which the checker cannot read into a \
                fixed set of premises"
                    .to_owned(),
            );
        };
        match name.as_str() {
            "stdin" => stdin_pipe = is_asyncio_subprocess_pipe(&keyword.value),
            "stdout" => stdout_pipe = is_asyncio_subprocess_pipe(&keyword.value),
            other => {
                return Some(format!(
                    "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                ));
            }
        }
    }
    if !stdin_pipe {
        return Some(
            "this call does not pass stdin=asyncio.subprocess.PIPE, so the checker cannot tell that the \
            payload crosses out on stdin"
                .to_owned(),
        );
    }
    if !stdout_pipe {
        return Some(
            "this call does not pass stdout=asyncio.subprocess.PIPE, so the checker cannot read the \
            target's stdout back"
                .to_owned(),
        );
    }
    None
}

/// Whether an expression is exactly `asyncio.subprocess.PIPE` — the
/// awaited shape's own spelling of the sync `subprocess.PIPE` sentinel
/// (`library/asyncio-subprocess.rst`: `asyncio.subprocess.PIPE` is the
/// same integer constant `subprocess.PIPE` is, re-exported under the
/// `asyncio.subprocess` namespace for this callee's own keywords). A
/// two-level attribute chain (`asyncio.subprocess.PIPE`), unlike the
/// sync shape's one-level `subprocess.PIPE`.
pub(super) fn is_asyncio_subprocess_pipe(expression: &Expr) -> bool {
    let Expr::Attribute(pipe_attribute) = expression else {
        return false;
    };
    if pipe_attribute.attr.as_str() != "PIPE" {
        return false;
    }
    let Expr::Attribute(subprocess_attribute) = pipe_attribute.value.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = subprocess_attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "asyncio" && subprocess_attribute.attr.as_str() == "subprocess"
}

/// Reads the `NamedTemporaryFile(...)` keyword arguments: `mode="w"`,
/// `suffix=".json"`, `delete=False` — ALL required, any other keyword
/// declines. `None` when every keyword checks out.
pub(super) fn temp_file_keywords_of(call_expr: &Expr) -> Option<String> {
    let Expr::Call(call) = call_expr else {
        return Some("this is not a call expression".to_owned());
    };
    let mut mode_w = false;
    let mut suffix_json = false;
    let mut delete_false = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return Some(
                "this call passes a keyword argument through **, which the checker cannot read into a \
                fixed set of premises"
                    .to_owned(),
            );
        };
        match name.as_str() {
            "mode" => mode_w = literal_string(&keyword.value) == Some("w"),
            "suffix" => suffix_json = literal_string(&keyword.value) == Some(".json"),
            "delete" => delete_false = matches!(&keyword.value, Expr::BooleanLiteral(literal) if !literal.value),
            other => {
                return Some(format!(
                    "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                ));
            }
        }
    }
    if !mode_w {
        return Some("this call does not pass mode=\"w\", so the checker cannot tell the handle is opened \
            for text writing"
            .to_owned());
    }
    if !suffix_json {
        return Some("this call does not pass suffix=\".json\", so the checker cannot tell the temp file is \
            named as JSON"
            .to_owned());
    }
    if !delete_false {
        return Some("this call does not pass delete=False, so the checker cannot tell the file survives \
            past the with-block for the call to read"
            .to_owned());
    }
    None
}

/// Reads the `subprocess.run` keyword arguments: `input=json.dumps(...)`,
/// `capture_output=True`, `text=True` — ALL required, any other keyword
/// declines. Answers the payload expression (`None` when `input` is
/// absent or is not a stringify-shaped call) and, on the FIRST keyword
/// shape that stops recognition, the decline sentence naming it.
pub(super) fn subprocess_run_keywords_of(call: &ExprCall) -> (Option<Expr>, Option<String>) {
    let mut payload: Option<Expr> = None;
    let mut capture_output_true = false;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return (
                None,
                Some("this call passes a keyword argument through **, which the checker cannot read \
                    into a fixed set of premises"
                    .to_owned()),
            );
        };
        match name.as_str() {
            "input" => {
                payload = json_dumps_argument_of(&keyword.value);
                if payload.is_none() {
                    return (
                        None,
                        Some("this call's input keyword is not json.dumps(...), so the checker cannot \
                            read what crosses out on stdin"
                            .to_owned()),
                    );
                }
            }
            "capture_output" => capture_output_true = literal_true(&keyword.value),
            "text" => text_true = literal_true(&keyword.value),
            other => {
                return (
                    None,
                    Some(format!(
                        "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                    )),
                );
            }
        }
    }
    if !capture_output_true {
        return (
            None,
            Some("this call does not pass capture_output=True, so the checker cannot read the target's \
                stdout back"
                .to_owned()),
        );
    }
    if !text_true {
        return (
            None,
            Some("this call does not pass text=True, so its result is bytes rather than the target's \
                JSON text — the return leg has no text to parse"
                .to_owned()),
        );
    }
    (payload, None)
}

/// Reads the `subprocess.run` keyword arguments for the argv-json call
/// shape: `capture_output=True` and `text=True` are required exactly as
/// they are for the stdin shape, but `input` is admitted here ONLY to
/// be detected and reported back — its presence alongside an argv
/// payload is the double-channel case the caller declines, never a
/// silent second reading of the payload. Any other keyword declines.
pub(super) fn subprocess_run_argv_json_keywords_of(call: &ExprCall) -> (bool, Option<String>) {
    let mut input_present = false;
    let mut capture_output_true = false;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return (
                false,
                Some("this call passes a keyword argument through **, which the checker cannot read \
                    into a fixed set of premises"
                    .to_owned()),
            );
        };
        match name.as_str() {
            "input" => input_present = true,
            "capture_output" => capture_output_true = literal_true(&keyword.value),
            "text" => text_true = literal_true(&keyword.value),
            other => {
                return (
                    false,
                    Some(format!(
                        "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                    )),
                );
            }
        }
    }
    if !capture_output_true {
        return (
            input_present,
            Some("this call does not pass capture_output=True, so the checker cannot read the target's \
                stdout back"
                .to_owned()),
        );
    }
    if !text_true {
        return (
            input_present,
            Some("this call does not pass text=True, so its result is bytes rather than the target's \
                JSON text — the return leg has no text to parse"
                .to_owned()),
        );
    }
    (input_present, None)
}

/// Reads the `subprocess.check_output` keyword arguments:
/// `input=json.dumps(...)` and `text=True` — both required, any other
/// keyword declines. `check_output` has no `capture_output` keyword at
/// all (the callee always captures — `library/subprocess.rst`), so it
/// is not read here.
pub(super) fn subprocess_check_output_keywords_of(call: &ExprCall) -> (Option<Expr>, Option<String>) {
    let mut payload: Option<Expr> = None;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return (
                None,
                Some("this call passes a keyword argument through **, which the checker cannot read \
                    into a fixed set of premises"
                    .to_owned()),
            );
        };
        match name.as_str() {
            "input" => {
                payload = json_dumps_argument_of(&keyword.value);
                if payload.is_none() {
                    return (
                        None,
                        Some("this call's input keyword is not json.dumps(...), so the checker cannot \
                            read what crosses out on stdin"
                            .to_owned()),
                    );
                }
            }
            "text" => text_true = literal_true(&keyword.value),
            other => {
                return (
                    None,
                    Some(format!(
                        "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                    )),
                );
            }
        }
    }
    if !text_true {
        return (
            None,
            Some("this call does not pass text=True, so its result is bytes rather than the target's \
                JSON text — the return leg has no text to parse"
                .to_owned()),
        );
    }
    (payload, None)
}

/// Reads the `subprocess.Popen` keyword arguments: `stdin=subprocess
/// .PIPE`, `stdout=subprocess.PIPE`, `text=True` — ALL required
/// (`.communicate()`'s own call, not this one, carries the payload), any
/// other keyword declines. Answers the decline sentence naming the
/// first shape that stops recognition, or `None` when every keyword
/// checks out.
pub(super) fn subprocess_popen_keywords_of(call: &ExprCall) -> Option<String> {
    let mut stdin_pipe = false;
    let mut stdout_pipe = false;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return Some(
                "this call passes a keyword argument through **, which the checker cannot read into a \
                fixed set of premises"
                    .to_owned(),
            );
        };
        match name.as_str() {
            "stdin" => stdin_pipe = is_subprocess_pipe(&keyword.value),
            "stdout" => stdout_pipe = is_subprocess_pipe(&keyword.value),
            "text" => text_true = literal_true(&keyword.value),
            other => {
                return Some(format!(
                    "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                ));
            }
        }
    }
    if !stdin_pipe {
        return Some(
            "this call does not pass stdin=subprocess.PIPE, so the checker cannot tell that the payload \
            crosses out on stdin"
                .to_owned(),
        );
    }
    if !stdout_pipe {
        return Some(
            "this call does not pass stdout=subprocess.PIPE, so the checker cannot read the target's \
            stdout back"
                .to_owned(),
        );
    }
    if !text_true {
        return Some(
            "this call does not pass text=True, so its result is bytes rather than the target's JSON \
            text — the return leg has no text to parse"
                .to_owned(),
        );
    }
    None
}

/// Whether an expression is exactly `subprocess.PIPE`.
pub(super) fn is_subprocess_pipe(expression: &Expr) -> bool {
    let Expr::Attribute(attribute) = expression else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "subprocess" && attribute.attr.as_str() == "PIPE"
}

/// Reads `json.dumps(<expr>)` and answers the single argument. An
/// f-string wrapping EXACTLY one interpolation, with no literal text
/// around it and no conversion (`!s`/`!r`/`!a`) or format spec
/// (`level_via_fstring_argv_data`'s own `f"{json.dumps(boosted)}"`
/// shape) unwraps to that interpolation's own inner expression first —
/// the spelling an f-string always produces for a lone substitution, and
/// the only shape `json_dumps_argument_of` reads through; a literal
/// character anywhere alongside the interpolation, more than one
/// interpolation, or a conversion/format spec all decline unchanged
/// (falls through to the `Expr::Call` check below, which then answers
/// `None` for the wrapper).
pub(super) fn json_dumps_argument_of(expression: &Expr) -> Option<Expr> {
    let expression = single_interpolation_call(expression).unwrap_or_else(|| expression.clone());
    let expression = &expression;
    let Expr::Call(call) = expression else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "json" || attribute.attr.as_str() != "dumps" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return None;
    };
    Some(argument.clone())
}

/// The sole interpolation's own inner expression, for an f-string that
/// spells EXACTLY one substitution and nothing else — no literal
/// character before, between, or after it, and no conversion or format
/// spec on that one interpolation. `None` for every other f-string shape
/// (mixed literal text, zero or multiple interpolations, a conversion or
/// format spec, or an implicitly concatenated f-string, which
/// `as_single_part_fstring` itself already declines) and for any
/// expression that is not `Expr::FString` at all.
pub(super) fn single_interpolation_call(expression: &Expr) -> Option<Expr> {
    let Expr::FString(fstring) = expression else {
        return None;
    };
    let single = fstring.as_single_part_fstring()?;
    let elements: &[InterpolatedStringElement] = &single.elements;
    let [InterpolatedStringElement::Interpolation(interpolation)] = elements else {
        return None;
    };
    if interpolation.conversion != ConversionFlag::None || interpolation.format_spec.is_some() {
        return None;
    }
    Some(interpolation.expression.as_ref().clone())
}

/// A written string literal's own text.
pub(super) fn literal_string(expression: &Expr) -> Option<&str> {
    match expression {
        Expr::StringLiteral(literal) => Some(literal.value.to_str()),
        _ => None,
    }
}

/// Whether an expression is the literal `True`. `pub(crate)`: shared with
/// `expressions.rs`'s own `subprocess.run(...).stdout` attribute-read
/// recognition (`stdout_attribute_of_recognized_run`), which reads the
/// same `capture_output=True`/`text=True` keyword shape this file's own
/// `subprocess_run_keywords_of` already checks, without pulling in the
/// full argv/payload/artifact machinery that function's job (proving a
/// crossing) actually needs.
pub(crate) fn literal_true(expression: &Expr) -> bool {
    matches!(expression, Expr::BooleanLiteral(literal) if literal.value)
}

