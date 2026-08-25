//! The serialization/encoding side: reading one named function's row —
//! its entry positions, its return, and its provenance — out of the
//! parsed JSON envelope, decoding every wire-carried set through the
//! kernel's own decoder under `catch_unwind`.

use refined_kernel::wire_decode::decode_wire_set;
use serde_json::Value;

use super::producer::FOREIGN_EXPORT_COMMAND;
use super::types::ForeignCase;
use super::types::ForeignTsEntry;
use super::types::ForeignTsFunctionFact;
use super::typescript_read::quoted_or_none;

/// Reads one named function's row: its entry positions, its return, and
/// the provenance a cross-language message renders. `decode_wire_set`
/// panics on a form it does not know — its own stated contract for
/// kernel answers. An artifact is a file another program wrote, so a
/// malformed form is a decline here, never a crash.
pub(super) fn function_fact_of(parsed: &Value, name: &str, artifact_path_words: &str) -> Result<ForeignTsFunctionFact, String> {
    let Some(functions) = parsed.get("functions").and_then(Value::as_object) else {
        return Err(format!("{artifact_path_words} carries no \"functions\" object at all"));
    };
    let Some(row) = functions.get(name).and_then(Value::as_object) else {
        return Err(format!(
            "{artifact_path_words} names {name} as the harness's called function, but \"functions\" carries no \
             row for it"
        ));
    };
    let decoded = crate::kernel_ask::ask_kernel(|| function_fact_of_row(row, name));
    match decoded {
        Ok(Ok(fact)) => Ok(fact),
        Ok(Err(sentence)) => Err(format!("{artifact_path_words} {sentence}")),
        Err(_) => Err(format!(
            "{artifact_path_words} states a set this checker's kernel grammar does not read, so the fact for \
             {name} cannot be decoded"
        )),
    }
}

/// One function row's fields read out, under the caller's `catch_unwind`
/// — every `decode_wire_set` call here can panic on a malformed form.
fn function_fact_of_row(row: &serde_json::Map<String, Value>, name: &str) -> Result<ForeignTsFunctionFact, String> {
    let entries = artifact_entries_of(row, name)?;
    let Some(returned) = row.get("return").and_then(Value::as_object) else {
        return Err(format!("carries no \"return\" object for {name}, so nothing crosses back from this call"));
    };
    let return_cases = cases_of(returned, &format!("the return for {name}"))?;
    let stdout_pure = returned.get("stdoutPure").and_then(Value::as_bool).unwrap_or(false);
    let (provenance_line, provenance_said) = artifact_provenance_of(row);
    Ok(ForeignTsFunctionFact {
        name: name.to_owned(),
        entry: entries,
        return_cases,
        stdout_pure,
        provenance_line,
        provenance_said,
    })
}

/// Reads the entry rows in the order the artifact spells them — that
/// order IS the positional order of the target's parameters, which is
/// how an argument finds the row it must fit.
fn artifact_entries_of(row: &serde_json::Map<String, Value>, name: &str) -> Result<Vec<ForeignTsEntry>, String> {
    let Some(raw_entries) = row.get("entry").and_then(Value::as_array) else {
        return Err(format!("states no entry positions for {name}, so nothing says what the target admits"));
    };
    let mut entries = Vec::with_capacity(raw_entries.len());
    for (index, raw_entry) in raw_entries.iter().enumerate() {
        let Some(entry_row) = raw_entry.as_object() else {
            return Err(format!("states an unreadable entry position {index} for {name}"));
        };
        let entry_name = entry_row.get("name").and_then(Value::as_str).unwrap_or("").to_owned();
        if let Some(sequence) = entry_row.get("sequence").and_then(Value::as_object) {
            let Some(element) = sequence.get("element").and_then(Value::as_object) else {
                return Err(format!("states a sequence entry {entry_name} for {name} with no element cases"));
            };
            let element_cases = cases_of(element, &format!("the sequence entry {entry_name} for {name}"))?;
            let length_at_least = sequence.get("lengthAtLeast").and_then(Value::as_i64).unwrap_or(0);
            entries.push(ForeignTsEntry {
                name: entry_name,
                sequence: Some((element_cases, length_at_least)),
                scalar: None,
            });
            continue;
        }
        let scalar_cases = cases_of(entry_row, &format!("the entry position {entry_name} for {name}"))?;
        entries.push(ForeignTsEntry {
            name: entry_name,
            sequence: None,
            scalar: Some(scalar_cases),
        });
    }
    Ok(entries)
}

/// Reads a `"cases"` array off an object that carries one — the RULED
/// schema's own unit at both the return position and a scalar entry
/// position (and a sequence entry's `element` object). Strict-parse: a
/// bare `"set"` field (the earlier shape) is NOT read as a one-case
/// fallback — that shape is exactly what the no-version-ceremony rule
/// calls NO-FACT, and the caller's own decline sentence is what a stale
/// artifact earns, never a silent best-effort reinterpretation.
fn cases_of(carrier: &serde_json::Map<String, Value>, described: &str) -> Result<Vec<ForeignCase>, String> {
    let Some(raw_cases) = carrier.get("cases").and_then(Value::as_array) else {
        return Err(format!(
            "{described} states no \"cases\" array, so nothing says what shape the value takes — re-export \
             it with `{FOREIGN_EXPORT_COMMAND} <target>`"
        ));
    };
    cases_array_of(raw_cases, described)
}

/// Reads a cases array directly — `cases_of`'s own body, factored out so
/// an object case's MEMBER (whose value in the wire IS the bare cases
/// array, `fact_export.rs::Case::to_json`'s own `Case::Object` arm: `{name:
/// cases_json(cases)}`, never a `{"cases": [...]}` wrapper) parses through
/// the identical rule rather than a second copy of it.
fn cases_array_of(raw_cases: &[Value], described: &str) -> Result<Vec<ForeignCase>, String> {
    if raw_cases.is_empty() {
        return Err(format!("{described} states an empty \"cases\" array, which admits no value at all"));
    }
    let mut cases = Vec::with_capacity(raw_cases.len());
    for (index, raw_case) in raw_cases.iter().enumerate() {
        let Some(case) = raw_case.as_object() else {
            return Err(format!("{described} states an unreadable case {index}"));
        };
        let sort = case.get("sort").and_then(Value::as_str).unwrap_or("");
        cases.push(match sort {
            "number" => {
                let Some(raw_set) = case.get("set") else {
                    return Err(format!("{described} states a number case {index} with no set"));
                };
                ForeignCase::Number(decode_wire_set(raw_set))
            }
            "string" => {
                let Some(raw_set) = case.get("set") else {
                    return Err(format!("{described} states a string case {index} with no set"));
                };
                ForeignCase::String(decode_wire_set(raw_set))
            }
            "boolean" => ForeignCase::Boolean,
            "null" => ForeignCase::Null,
            "object" => object_case_of(case, &format!("{described}'s case {index}"))?,
            other => {
                return Err(format!(
                    "{described} states a case {index} of sort {}, and this reader admits only \"number\", \
                     \"string\", \"boolean\", \"null\", or \"object\"",
                    quoted_or_none(other)
                ));
            }
        });
    }
    Ok(cases)
}

/// Reads one `{"sort": "object", "members": {...}, "closed": bool}` case
/// — the RULED object case's own strict parse (CROSS-LANGUAGE-EDGE.md
/// §17, JT-prioritized 2026-08-21). `members` must be a JSON OBJECT
/// mapping each key DIRECTLY to its own cases ARRAY (never a `{"cases":
/// [...]}` wrapper — `fact_export.rs::Case::to_json`'s `Case::Object` arm
/// writes `{name: cases_json(cases)}`, the bare array, so the parser's
/// shape must match the writer's exactly), recursed through
/// `cases_array_of` so a nested object case parses through the identical
/// rule; `closed` must be a JSON BOOLEAN. Any deviation — `members`
/// missing or not an object, `closed` missing or not a boolean, a
/// member's own value not itself a cases array — declines by name through
/// the ordinary `Err` path, exactly the same "an artifact is a file, not
/// a promise" discipline every other malformed shape in this file earns;
/// nothing here guesses at a member.
fn object_case_of(case: &serde_json::Map<String, Value>, described: &str) -> Result<ForeignCase, String> {
    let Some(raw_members) = case.get("members").and_then(Value::as_object) else {
        return Err(format!(
            "{described} states an object case with no \"members\" object, so nothing says what keys it admits"
        ));
    };
    let Some(closed) = case.get("closed").and_then(Value::as_bool) else {
        return Err(format!(
            "{described} states an object case with no boolean \"closed\" field, so nothing says whether its \
             key set is exact"
        ));
    };
    let mut members = Vec::with_capacity(raw_members.len());
    for (name, raw_member) in raw_members {
        let Some(member_cases) = raw_member.as_array() else {
            return Err(format!("{described} states a member '{name}' that is not a cases array"));
        };
        let cases = cases_array_of(member_cases, &format!("{described}'s member '{name}'"))?;
        members.push((name.clone(), cases));
    }
    Ok(ForeignCase::Object { members, closed })
}

/// Reads where the target's claim was made. Absent fields answer
/// `(0, "")` rather than declining — provenance makes a message
/// readable; it is not a premise of the crossing.
fn artifact_provenance_of(row: &serde_json::Map<String, Value>) -> (usize, String) {
    let Some(provenance) = row.get("provenance").and_then(Value::as_object) else {
        return (0, String::new());
    };
    let line = provenance.get("line").and_then(Value::as_i64).unwrap_or(0).max(0) as usize;
    let said = provenance.get("said").and_then(Value::as_str).unwrap_or("").to_owned();
    (line, said)
}
