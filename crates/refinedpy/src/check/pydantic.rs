use refined_domain::abstract_value::{known_values, AbstractValue, ObjectKey, PrimitiveKind};
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustSpec;

/// pydantic's own ISO-8601 duration/datetime parse — `TypeAdapter(<a
/// temporal alias>).validate_python(<string>)`'s own reading, ahead of
/// `judge`'s ordinary scalar path (`adapter_alias_verdict`'s own call
/// site). `chart` names which of the two grammars applies (a
/// `PlainDate`-chart alias never reaches this function — a `date` base
/// carries no separate string parse row this crate models beyond
/// `date.fromisoformat`, which `expressions.rs` already owns for the
/// CONSTRUCTOR-call shape).
///
/// `Duration`: pydantic's own month/year substitution — AGENT-BRIEF.md's
/// pydantic-surface-facts, execution-verified against pydantic 2.13.4:
/// "`timedelta` fields parse ISO 8601 duration strings ... Month/year
/// designators are NOT rejected: pydantic-core substitutes fixed spans
/// (`P1M` → exactly 30 days, `P1Y` → exactly 365 days)." Read here by
/// rewriting a `PnY`/`PnM` designator to its own fixed-day equivalent
/// BEFORE the text ever reaches the kernel's own calendar seam (the
/// kernel's `duration_fields` reads Y/M as their own separate raw
/// components — the standard ISO reading, not pydantic's substitution —
/// so the rewrite must happen here, not there). Any other duration
/// shape (weeks/days/hours/minutes/seconds, fractional seconds) is
/// pydantic-core's ordinary ISO-8601 duration grammar, read UNCHANGED —
/// the same grammar `refined_sets::calendar_interpreter::duration_
/// fields` already parses, so passing the text straight through lets
/// the kernel's own asks decide the rest (`FollowUp`'s own microsecond-
/// resolution rows). A string that fails EVEN the loose recognition
/// this function applies (no leading `P`/`p`) is not a duration at all
/// — pydantic-core RAISES on it (showcase.py's own `"not a duration"`
/// row) — read here as a GRAMMAR FIRE: the instance built carries no
/// `.temporal` at all, which `judge`'s own temporal law reads as "not a
/// temporal construction" (`is_temporal_construction` false) and
/// answers Undetermined — the caller's own `judge` call still reports
/// SOMETHING, but not a designated Fire this function does not itself
/// decide grammar-invalid text against; `pydantic_duration_grammar_
/// fire` below is what actually fires it, checked by the caller ahead
/// of this function's own parse.
///
/// `Instant`: pydantic's own `datetime` parse (execution-verified: an
/// offset-bearing ISO string parses aware at that exact offset; an
/// offset-free string parses NAIVE — `AwareDatetime`'s own refusal of
/// it is `assignability.rs`'s temporal admission law, not this
/// function's concern). RFC 9557 `[Zone/Name]` bracket suffixes do NOT
/// parse under stock pydantic (AGENT-BRIEF.md, same section) — read
/// here as a decline (`None`), the same "this table cannot read it"
/// answer any other unrecognized shape gives.
pub(super) fn pydantic_temporal_parse(text: &str, chart: refined_sets::calendar_interpreter::TemporalChart) -> Option<AbstractValue> {
    use refined_sets::calendar_interpreter::TemporalChart;
    match chart {
        TemporalChart::Duration => pydantic_duration_value(text),
        TemporalChart::Instant => pydantic_datetime_string_value(text),
        _ => None,
    }
}

/// pydantic's own `PnY`/`PnM` substitution, applied to the ISO text
/// BEFORE the kernel ever sees it — `pydantic_temporal_parse`'s own
/// doc. Every OTHER duration shape (days/weeks/time fields, or no
/// Y/M designator at all) is returned as pydantic-core's own grammar,
/// unrewritten. `None` when the text does not even open with a `P`/`p`
/// sign-prefixed designator (`DURATION_SIGN_PREFIX_RE`'s own shape,
/// mirrored here) — not a duration at all, the grammar-fire case this
/// function's own caller checks separately.
pub(super) fn pydantic_duration_days_normalized(text: &str) -> Option<String> {
    let (sign, rest) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", text.strip_prefix('+').unwrap_or(text)),
    };
    let rest = rest.strip_prefix('P').or_else(|| rest.strip_prefix('p'))?;
    // A bare `PnY` or `PnM` (an all-digit count immediately followed by
    // exactly one of `Y`/`y`/`M`/`m`, and nothing else) — the one shape
    // pydantic substitutes; every other duration (`P30D`, `P1DT2H`, a
    // combined `P1Y2M3D`, …) is returned unchanged, since pydantic-core's
    // own substitution is documented only for the bare single-designator
    // form this crate's own showcase row (`P2M`) exercises.
    let (count_str, unit) = rest.split_at(rest.len().saturating_sub(1));
    if count_str.is_empty() || !count_str.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let count: i64 = count_str.parse().ok()?;
    let days = match unit {
        "Y" | "y" => count.checked_mul(365)?,
        "M" | "m" => count.checked_mul(30)?,
        _ => return None,
    };
    Some(format!("{sign}P{days}D"))
}

/// `TypeAdapter(<Duration-chart alias>).validate_python(<text>)` — a
/// tagged `datetime_timedelta` `Kind::Object` carrying the normalized
/// ISO text as its own `.temporal` window (`Duration` chart, both
/// `min`/`max` the same point — an exact value, not a range), the same
/// shape `expressions.rs::timedelta_construction_value` builds for a
/// `timedelta(days=n)` CONSTRUCTOR call. `None` when the text is not a
/// Y/M-substitutable duration (`pydantic_duration_days_normalized`
/// declines) — the ORIGINAL text still rides through, unrewritten,
/// since a `P30D`/`P1DT2H` shape needs no substitution and is still a
/// duration the kernel's own `duration_fields` reads directly.
pub(super) fn pydantic_duration_value(text: &str) -> Option<AbstractValue> {
    let normalized = pydantic_duration_days_normalized(text).unwrap_or_else(|| text.to_owned());
    if !normalized.starts_with('P') && !normalized.starts_with('p') && !normalized.starts_with('-') && !normalized.starts_with('+') {
        return None;
    }
    let mut instance = known_object(Vec::new(), None, true, TrustSpec, false);
    instance.source = "datetime_timedelta".to_owned();
    instance.temporal = Some(Box::new(refined_sets::calendar_interpreter::TemporalAnnotation {
        chart: refined_sets::calendar_interpreter::TemporalChart::Duration,
        min: Some(normalized.clone()),
        max: Some(normalized),
    }));
    Some(instance)
}

/// `TypeAdapter(<Instant-chart alias>).validate_python(<text>)` — a
/// tagged `datetime_datetime` `Kind::Object` carrying the text AS ITS
/// OWN ISO spelling directly (pydantic's own `datetime` string parse
/// accepts the same grammar `calendar_interpreter.rs`'s own `Instant`
/// chart reader does — an offset-bearing string parses aware at that
/// exact offset, an offset-free string parses naive). `instance.aware`
/// is read from the TEXT's own trailing offset marker (`Z`/`+HH:MM`/
/// `-HH:MM` — a bare offset spelling, never `Z` folded to a named
/// zone), matching `assignability.rs`'s own `aware`-field admission
/// law: `0` naive, `1` aware (this reader never distinguishes UTC from
/// a non-UTC exact offset the way `expressions.rs`'s own construction
/// path does — a parsed string with ANY offset is `1`, since pydantic
/// itself resolves the exact instant from the string's own offset
/// regardless of whether that offset happens to be zero). A
/// zone-bracket suffix (`[Zone/Name]`) declines (`None`) — RFC 9557
/// does not parse under stock pydantic (`pydantic_temporal_parse`'s
/// own doc).
pub(super) fn pydantic_datetime_string_value(text: &str) -> Option<AbstractValue> {
    if text.contains('[') {
        return None;
    }
    let has_offset = text.ends_with('Z')
        || text.ends_with('z')
        || text[1..].contains('+')
        || text[1..].rfind('-').is_some_and(|index| index > 9);
    let mut instance = known_object(
        vec![ObjectKey {
            name: "aware".to_owned(),
            numeric: false,
            value: known_values(vec![if has_offset { 1.0 } else { 0.0 }], PrimitiveKind::Integer, TrustSpec),
        }],
        None,
        true,
        TrustSpec,
        false,
    );
    instance.source = "datetime_datetime".to_owned();
    instance.temporal = Some(Box::new(refined_sets::calendar_interpreter::TemporalAnnotation {
        chart: refined_sets::calendar_interpreter::TemporalChart::Instant,
        min: Some(text.to_owned()),
        max: Some(text.to_owned()),
    }));
    Some(instance)
}
