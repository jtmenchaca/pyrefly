use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::diagnostic_sentences;
use crate::env::Environment;

use super::super::argv::RecognitionDecline;
use super::super::argv::Runner;
use super::super::argv::resolve_target_path;
use super::super::argv::script_extension_decline;
use super::super::argv::split_runner_and_script_tagged;
use super::super::argv::tokenize_shell_command;
use super::super::keywords::literal_string;
use super::super::parse_consumer::visit_statement_exprs;
use super::super::Channel;
use super::super::ForeignEdge;
use super::super::ResultRead;
use super::as_bare_name;
use super::json_dump_payload_of;

/// `<name> = os.system("<shell command>")` — the runner, script, and any
/// `< infile`/`> outfile` redirections tokenize the same way regardless
/// of what the redirections turn out to mean. A command carrying BOTH
/// redirections is tried as a full crossing first (`recognize_os_system_
/// file_legs` — ONE-CHECKER.md item 2's own current text: the invocation's
/// OWN data legs, read like stdin/captured-stdout, never item 3's
/// unlinked data-fact machinery); that reader answers its own decline
/// naming whichever piece (the entry write, the return read) it could not
/// find, so this function returns whatever it answers unchanged. Every
/// OTHER shape (missing command, unreadable string, an unrecognized
/// runner, a redirection missing one side, or an unsupported trailing
/// token) still declines here exactly as before: `os.system` names a
/// runner/script/file set this checker can read, but only the file-legs
/// shape above ever turns that reading into a value-carrying crossing.
///
/// `None` when this is not an `os.system` call at all (no sentence
/// owed) or the module name is shadowed, mirroring the `subprocess`
/// shadow-on-rebind check every other recognizer applies.
pub(super) fn recognize_os_system(
    statements: &[Stmt],
    index: usize,
    assign: &StmtAssign,
    environment: &Environment,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let Expr::Call(call) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "os" || environment.read("os").is_some() {
        return None;
    }
    if attribute.attr.as_str() != "system" {
        return None;
    }
    let call_range = call.range();
    let [command] = call.arguments.args.as_ref() else {
        return Some(Err(RecognitionDecline {
            message: "this call passes other than one positional command argument, and the checker \
                models only a single written shell-string argument"
                .to_owned(),
            range: call_range,
        }));
    };
    let Some(command_text) = literal_string(command) else {
        return Some(Err(RecognitionDecline {
            message: diagnostic_sentences::os_system_shell_string_unreadable(),
            range: call_range,
        }));
    };
    let Some(tokens) = tokenize_shell_command(command_text) else {
        return Some(Err(RecognitionDecline {
            message: diagnostic_sentences::os_system_shell_string_unreadable(),
            range: call_range,
        }));
    };
    let Some((runner, runner_and_script, remainder)) = split_runner_and_script_tagged(&tokens) else {
        return Some(Err(RecognitionDecline {
            message: diagnostic_sentences::os_system_shell_string_unreadable(),
            range: call_range,
        }));
    };
    // "< infile" and "> outfile" are the two redirections this row reads
    // past the runner and script, in either order or both — a command
    // line's own way of naming stdin/stdout files. Any other trailing
    // token is unsupported and named specifically rather than silently
    // accepted.
    let Some((infile, outfile)) = redirection_files_of(remainder) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "{} is followed by {}, which this checker's shell-string reader does not admit — only \
                trailing \"< <file>\"/\"> <file>\" redirections are read past the runner and script",
                runner_and_script,
                remainder.join(" ")
            ),
            range: call_range,
        }));
    };
    let (Some(infile), Some(outfile)) = (infile, outfile) else {
        // Fewer than both redirections: the runner and script (and
        // whichever one redirection is present) still read cleanly, but
        // there is no full file-legs shape to try — the same
        // no-stdout-capture decline as before, naming whatever the
        // command's own text spelled.
        let redirection_suffix = redirection_suffix_of(remainder).unwrap_or_default();
        return Some(Err(RecognitionDecline {
            message: diagnostic_sentences::os_system_no_stdout_capture(&(runner_and_script + &redirection_suffix)),
            range: call_range,
        }));
    };
    recognize_os_system_file_legs(statements, index, call_range, runner, &runner_and_script, infile, outfile)
}

/// `os.system("<runner> <script> < <infile> > <outfile>")` with both
/// redirections present: ONE-CHECKER.md item 2's own current text — the
/// command string deterministically names the code AND both files in
/// one recognized reference, so the redirections are the invocation's
/// OWN data legs, read like stdin/captured-stdout rather than item 3's
/// unlinked data-fact machinery. The entry leg binds from a PRECEDING
/// `with open("<infile>", "w") as <handle>: json.dump(<payload>,
/// <handle>)` in the SAME body (`os_system_entry_write_of`), checked
/// HERE, before any artifact lookup — the same ordering every other
/// recognizer keeps between reading the call's own shape and discharging
/// the artifact's premises. The return leg binds at a LATER `with
/// open("<outfile>") as <handle>: ... json.load(<handle>)` in the same
/// body, but that scan runs AFTER the call statement — `result_read:
/// ResultRead::FileRead { outfile }` carries the literal outfile name
/// forward so `finish_recognized_edge` can run `os_system_return_read_of`
/// once `discharge_edge_premises` has already passed. Either leg missing
/// keeps a named decline (this one here for the entry leg; `finish_
/// recognized_edge`'s own `Blocked` for the return leg) — this row is
/// read whole or not at all, never a partial crossing.
///
/// Builds a `ForeignEdge` with `channel: Channel::Stdin` (the same
/// carrier `subprocess.run`'s own stdin-json shape uses: a redirected
/// file's bytes land on the SAME fd a pipe would, and `level_ok.ts`'s
/// own harness reads stdin either way) so `discharge_edge_premises`
/// discharges the runtime-identity, carrier-identity, outbound-leg, and
/// channel-purity premises exactly as it does for every other recognized
/// shape — the ONE leg-judging path this row never duplicates.
/// `result_name` carries no meaning for this shape (there is no bound
/// name the return leg reads through) and stays empty.
pub(super) fn recognize_os_system_file_legs(
    statements: &[Stmt],
    index: usize,
    call_range: TextRange,
    runner: Runner,
    runner_and_script: &str,
    infile: &str,
    outfile: &str,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    // the script itself must still be a `.ts` file this checker models a
    // fact for, exactly as every other recognized shape requires
    let script_text = runner_and_script
        .rsplit(' ')
        .next()
        .expect("split_runner_and_script_tagged always leaves the script as the last token")
        .to_owned();
    if let Some(decline) = script_extension_decline(&script_text, runner, call_range) {
        return Some(Err(decline));
    }
    let Some(payload) = os_system_entry_write_of(statements, index, infile) else {
        return Some(Err(RecognitionDecline {
            message: diagnostic_sentences::os_system_missing_entry_write(infile),
            range: call_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&script_text),
        payload,
        channel: Channel::Stdin,
        result_name: String::new(),
        result_read: ResultRead::FileRead { outfile: outfile.to_owned() },
        consumer_scan_from: index,
        runner,
    }))
}

/// Scans BACKWARD over `statements[..index]` for the LAST `with
/// open("<infile>", "w"[, ...]) as <handle>: json.dump(<payload>,
/// <handle>)` whose own literal filename matches `infile` exactly — the
/// entry leg's payload. The LAST such write, not the first: a body may
/// write `infile` more than once before the call reads it, and only the
/// write closest to the call is provably still on disk when `os.system`
/// runs. `None` when no such write exists anywhere before the call — the
/// caller's own named decline.
pub(super) fn os_system_entry_write_of(statements: &[Stmt], index: usize, infile: &str) -> Option<Expr> {
    statements[..index].iter().rev().find_map(|statement| {
        let Stmt::With(with_stmt) = statement else {
            return None;
        };
        let [item] = with_stmt.items.as_slice() else {
            return None;
        };
        if literal_open_call_names(&item.context_expr, &["w"]) != Some(infile) {
            return None;
        }
        let handle_name = item.optional_vars.as_deref().and_then(as_bare_name)?;
        let [dump_statement] = with_stmt.body.as_slice() else {
            return None;
        };
        json_dump_payload_of(dump_statement, handle_name)
    })
}

/// Scans FORWARD over `statements[index + 1..]` for the FIRST `with
/// open("<outfile>") as <handle>: ... json.load(<handle>)` whose own
/// literal filename matches `outfile` exactly — the return leg's
/// consumer node. `None` when no such read exists anywhere after the
/// call — the caller's own named decline. The `with`-block's body may
/// hold the `json.load(...)` node directly as a `return` statement (this
/// corpus's own row) or as a plain expression-statement assignment; both
/// read through the same `visit_statement_exprs`/`is_json_load_of` scan
/// this file already drives for every other return-leg search, applied
/// here to the WITH-BLOCK'S OWN body rather than the outer body.
pub(in crate::foreign_edge) fn os_system_return_read_of(statements: &[Stmt], index: usize, outfile: &str) -> Option<TextRange> {
    statements[index + 1..].iter().find_map(|statement| {
        let Stmt::With(with_stmt) = statement else {
            return None;
        };
        let [item] = with_stmt.items.as_slice() else {
            return None;
        };
        if literal_open_call_names(&item.context_expr, &[]) != Some(outfile) {
            return None;
        }
        let mut found = None;
        for inner in &with_stmt.body {
            visit_statement_exprs(inner, &mut |expression| {
                if found.is_none() && is_json_load_call(expression) {
                    found = Some(expression.range());
                }
            });
        }
        found
    })
}

/// Whether `expression` is exactly `open(<literal filename>[, <literal
/// mode>])` — a bare builtin call, never `module.open(...)` (`open` is
/// never shadowed by an attribute access this reader would need to rule
/// out, matching `library/functions.rst`'s own builtin-namespace
/// placement). `required_mode`, when non-empty, is the exact single
/// positional mode string this call must ALSO pass (`["w"]` for the
/// entry leg's own write-mode requirement; `[]` for the return leg's
/// bare `open(<file>)`, whose implicit default mode is `"r"`, matching
/// `library/functions.rst`, `open()`: "mode... defaults to 'r'"). `None`
/// — not this shape, or the mode does not match.
pub(super) fn literal_open_call_names<'a>(expression: &'a Expr, required_mode: &[&str]) -> Option<&'a str> {
    let Expr::Call(call) = expression else {
        return None;
    };
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "open" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    match (call.arguments.args.as_ref(), required_mode) {
        ([filename], []) => literal_string(filename),
        ([filename, mode], [expected_mode]) => {
            if literal_string(mode) != Some(expected_mode) {
                return None;
            }
            literal_string(filename)
        }
        _ => None,
    }
}

/// Whether a node is exactly `json.load(<any expression>)` — the plain
/// file-handle read `json.loads` has no equivalent for (`library/json
/// .rst`: `load` "deserialize... a text file", `loads` "deserialize... a
/// str"), so this reads the RECEIVER shape only (a bare positional
/// argument, no keywords), never checking which name it names: the
/// caller (`os_system_return_read_of`) already scoped the search to one
/// specific `with`-block's own body, so any `json.load(...)` reached
/// there reads THAT block's own handle by construction — the with-block
/// binds exactly one name, and `json.load`'s own argument has nowhere
/// else to come from inside that body.
pub(super) fn is_json_load_call(expression: &Expr) -> bool {
    let Expr::Call(call) = expression else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "json"
        && attribute.attr.as_str() == "load"
        && call.arguments.keywords.is_empty()
        && call.arguments.args.len() == 1
}

/// Reads zero, one, or both of a trailing `< infile` / `> outfile`
/// redirection, in either order, off the tokens following the runner
/// and script. `None` when the trailing tokens are not exactly this
/// shape (an extra flag, a pipe, anything this reader does not admit).
pub(super) fn redirection_suffix_of(remainder: &[&str]) -> Option<String> {
    match remainder {
        [] => Some(String::new()),
        ["<", input_file] => Some(format!(" < {input_file}")),
        [">", output_file] => Some(format!(" > {output_file}")),
        ["<", input_file, ">", output_file] => Some(format!(" < {input_file} > {output_file}")),
        [">", output_file, "<", input_file] => Some(format!(" > {output_file} < {input_file}")),
        _ => None,
    }
}

/// `redirection_suffix_of`'s own exact twin, answering the two file NAMES
/// themselves (`(infile, outfile)`, either possibly absent) rather than a
/// formatted decline suffix — what `recognize_os_system_file_legs` reads
/// to find the entry write and the return read this command's own
/// redirections name. Shares the identical match shape so the two readers
/// can never disagree about which trailing tokens are this shape at all.
pub(super) fn redirection_files_of<'a>(remainder: &'a [&'a str]) -> Option<(Option<&'a str>, Option<&'a str>)> {
    match remainder {
        [] => Some((None, None)),
        ["<", input_file] => Some((Some(input_file), None)),
        [">", output_file] => Some((None, Some(output_file))),
        ["<", input_file, ">", output_file] => Some((Some(input_file), Some(output_file))),
        [">", output_file, "<", input_file] => Some((Some(input_file), Some(output_file))),
        _ => None,
    }
}
