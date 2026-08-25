//! The `os.system`/cross-process-edge wording: the `os.system` argument-
//! shape declines (`os_system_shell_string_unreadable`, `script_path_
//! not_a_literal`, the missing-capture and missing-entry/return-file
//! declines), the compiled-binary decline, every channel-mismatch
//! sentence pairing two named carriers, the double-channel decline, and
//! `foreign_crossing_refusal`'s own two-language explanation.

/// `os.system`'s own undetermined reason for a command whose runner and
/// script DID read cleanly: even a followed literal shell command has
/// no value channel, because `os.system` never captures stdout — names
/// both the missing captured-stdout leg and the fixable argv-list
/// respelling.
pub fn os_system_no_stdout_capture(runner_and_script: &str) -> String {
    format!(
        "{runner_and_script} runs, but os.system captures no stdout — there is no captured-stdout leg for a \
        return fact to attach to; spell the call as subprocess.run([...], input=..., capture_output=True, \
        text=True) instead"
    )
}

/// The shell-string law-2 decline: `os.system`'s argument is not one
/// written string literal, so its tokens cannot be read at all.
pub fn os_system_shell_string_unreadable() -> String {
    "this command is a shell string the checker cannot read; spell it as an argv list \
    (subprocess.run([\"node\", \"<script>.ts\"], ...))"
        .to_owned()
}

/// The law-2 decline for a script path that is neither a written string
/// literal directly in the argv list nor a module-level constant this
/// body can resolve (an f-string, a parameter, any other computed
/// expression) — reused by every remaining non-literal argv[1] shape.
pub fn script_path_not_a_literal() -> String {
    "the script path is computed; spell it as a written string literal".to_owned()
}

/// A single-element, path-shaped argv (`["./targets/cpp_level"]`) names
/// the code that runs next — the compiled binary at that path — so the
/// recognizer reaches the artifact lookup exactly as a `node`/`deno`/
/// `bun`/`npx tsx` row does. No producer in this checker regenerates a
/// compiled binary's fact (`foreign_edge_artifact.rs::read_compiled_
/// binary_fact`'s own doc: no source this checker reads, no producer
/// binary), so the checker looks for a SIBLING fact file at
/// `<binary_path>.facts.json`, hand- or tool-authored; this sentence
/// names that construct when the sibling file is absent, rather than
/// the generic "there is no <path>.refined.json; write it with
/// -export-fact" sentence, which names a command that has no meaning
/// for a target that is not TypeScript source.
pub fn compiled_binary_no_fact(target_path: &str) -> String {
    format!(
        "{target_path} is a compiled binary, and there is no {target_path}.facts.json beside it — the \
        checker can name the code that runs next but has no fact stating what it does"
    )
}

/// `os.system`'s file-legs decline when the redirected IN-FILE has no
/// recognized write preceding the call in the same body: the runner,
/// script, and both redirections read cleanly, but there is no
/// `with open("<infile>", "w") as <handle>: json.dump(<payload>, <handle>)`
/// this checker can find, so no entry fact has anything to attach to.
pub fn os_system_missing_entry_write(infile: &str) -> String {
    format!(
        "this call redirects stdin from {infile}, but no `with open(\"{infile}\", \"w\") as <name>: \
        json.dump(<payload>, <name>)` precedes it in this body — the checker cannot find the value written \
        to the in-file"
    )
}

/// `os.system`'s file-legs decline when the redirected OUT-FILE has no
/// recognized read following the call in the same body: the runner,
/// script, both redirections, and the entry write all read cleanly, but
/// there is no `with open("<outfile>") as <handle>: ... json.load(<handle>)`
/// this checker can find, so the return leg has no consumer to attach a
/// fact to.
pub fn os_system_missing_return_read(outfile: &str) -> String {
    format!(
        "this call redirects stdout to {outfile}, but no `with open(\"{outfile}\") as <name>: ... \
        json.load(<name>)` follows it in this body — the checker cannot find the value read back from the \
        out-file"
    )
}

/// The channel-mismatch decline when the call sends its payload on
/// stdin but the target's own fact serves JSON on an argv element — the
/// reverse of `foreign_edge_channel_mismatch_argv_at_stdin_target`.
/// Neither side is malformed; the two simply do not name the same
/// carrier, so the JSON transport model has nothing to apply to.
pub fn foreign_edge_channel_mismatch_stdin_at_argv_target() -> String {
    "the call passes the payload on stdin, but the target's fact serves JSON on an argv element — the \
    channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when the call sends its payload as an
/// argv element but the target's own fact serves JSON on stdin.
pub fn foreign_edge_channel_mismatch_argv_at_stdin_target() -> String {
    "the call passes the payload as an argv element, but the target's fact serves JSON on stdin — the \
    channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when both sides name an argv carrier
/// but at different indices — the call writes the payload at one
/// position and the target reads it from another.
pub fn foreign_edge_channel_mismatch_argv_index(called_index: i64, declared_index: i64) -> String {
    format!(
        "the call passes the payload at argv[{called_index}], but the target's fact reads its payload at \
        argv[{declared_index}] — the channels do not meet"
    )
}

/// The channel-mismatch decline when the call writes a temp file named
/// at an argv element but the target's own fact serves JSON on stdin —
/// the file carrier's own mismatch against `stdin-json`, symmetric with
/// `foreign_edge_channel_mismatch_argv_at_stdin_target`.
pub fn foreign_edge_channel_mismatch_file_at_stdin_target() -> String {
    "the call passes the payload through a temp file named as an argv element, but the target's fact serves \
    JSON on stdin — the channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when the call sends its payload on
/// stdin but the target's own fact reads its JSON from a FILE named at
/// an argv element — the reverse of
/// `foreign_edge_channel_mismatch_file_at_stdin_target`.
pub fn foreign_edge_channel_mismatch_stdin_at_file_target() -> String {
    "the call passes the payload on stdin, but the target's fact reads its JSON from a file named as an \
    argv element — the channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when the call writes a temp file named
/// at an argv element but the target's own fact reads its argv element
/// AS the JSON text directly (`argv-json`), never as a file path.
pub fn foreign_edge_channel_mismatch_file_at_argv_target() -> String {
    "the call passes the payload through a temp file named as an argv element, but the target's fact reads \
    that argv element as the JSON text itself — the channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when the call passes its payload
/// directly as an argv element (`argv-json`), but the target's own fact
/// reads that argv element as a FILE PATH (`file-json`) rather than the
/// JSON text itself.
pub fn foreign_edge_channel_mismatch_argv_at_file_target() -> String {
    "the call passes the payload directly as an argv element, but the target's fact reads that argv element \
    as a file path holding the JSON — the channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when both sides name a file-carried argv
/// element but at different indices.
pub fn foreign_edge_channel_mismatch_file_index(called_index: i64, declared_index: i64) -> String {
    format!(
        "the call names the temp file at argv[{called_index}], but the target's fact reads the file's path \
        from argv[{declared_index}] — the channels do not meet"
    )
}

/// The double-channel decline when a call names BOTH an argv-json
/// payload and an `input=` keyword — a real ambiguity, not a recognition
/// gap: two crossing values are stated and this checker names one
/// transport per call.
pub fn foreign_edge_double_channel_declared() -> String {
    "this call passes the payload both as an argv element and through input=json.dumps(...) — two crossing \
    channels are named and this checker recognizes exactly one transport per call"
        .to_owned()
}

/// A cross-language crossing's refutation: the reason the value cannot
/// cross, with the target's own provenance appended — the second step
/// of the two-language explanation, in the message-text form
/// `foreign_edge.go`'s own `foreignMessage` renders. `provenance_line`
/// of 0 means the target stated no line (the provenance is present but
/// unlocated); an empty `provenance_said` alongside a nonzero line
/// still names WHERE without a quoted claim.
pub fn foreign_crossing_refusal(
    said: &str,
    provenance_file: &str,
    provenance_line: usize,
    provenance_said: &str,
) -> String {
    if provenance_file.is_empty() {
        return said.to_owned();
    }
    let mut where_said = provenance_file.to_owned();
    if provenance_line > 0 {
        where_said.push(':');
        where_said.push_str(&provenance_line.to_string());
    }
    if provenance_said.is_empty() {
        return format!("{said}. the target states this at {where_said}");
    }
    format!("{said}. {where_said} said: {provenance_said}")
}
