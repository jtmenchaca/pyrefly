//! A TZif (RFC 8536) reader over the system's compiled zoneinfo files
//! (`/usr/share/zoneinfo/<Zone/Name>`) — the same binary tzdata format
//! CPython's `zoneinfo` module reads. This crate has no timezone
//! dependency in either Cargo workspace (`pyrefly/Cargo.lock` and
//! `rust/Cargo.lock` both carry no `chrono-tz`/`jiff`/`tz-rs`), so a
//! literal instant in a literal zone name (`ZoneInfo("Europe/Paris")`)
//! is resolved by reading tzdata directly rather than adding a crate.
//!
//! Only what `expressions.rs::classify_tzinfo_expression` needs is
//! implemented: `utc_offset_seconds_for_wall_time`, the UTC offset in
//! effect for one LOCAL (wall-clock) instant in one named zone.

use std::fs;
use std::path::PathBuf;

/// One "local time type" record — a zone's own vocabulary entry for an
/// offset (`utoff`, seconds east of UTC) plus whether it is DST.
#[derive(Debug, Clone, Copy)]
struct LocalTimeType {
    utoff_seconds: i64,
    is_dst: bool,
}

/// One parsed TZif file: the transition instants (UTC seconds since
/// the epoch, ascending) with the local-time-type index each one
/// switches TO, plus the type table itself. `transitions[i]` is in
/// effect from `transition_at[i]` (inclusive) up to `transition_at[i+1]`
/// (exclusive); before the first transition, the first type whose
/// `is_dst` is false is the pre-transition default (RFC 8536 §3.2),
/// falling back to type 0 when every type is DST.
struct TzFile {
    transition_at: Vec<i64>,
    transition_type: Vec<u8>,
    types: Vec<LocalTimeType>,
}

/// Reads a big-endian u32 at `offset`, and returns the offset just
/// past it. `None` when the read would run past the buffer.
fn read_u32(bytes: &[u8], offset: usize) -> Option<(u32, usize)> {
    let slice = bytes.get(offset..offset + 4)?;
    Some((u32::from_be_bytes(slice.try_into().ok()?), offset + 4))
}

/// Reads a big-endian i64 at `offset` (the v2/v3 64-bit transition
/// time width), and returns the offset just past it.
fn read_i64(bytes: &[u8], offset: usize) -> Option<(i64, usize)> {
    let slice = bytes.get(offset..offset + 8)?;
    Some((i64::from_be_bytes(slice.try_into().ok()?), offset + 8))
}

/// Reads a big-endian i32 at `offset` (the local-time-type `utoff`
/// field's own width), and returns the offset just past it.
fn read_i32(bytes: &[u8], offset: usize) -> Option<(i32, usize)> {
    let slice = bytes.get(offset..offset + 4)?;
    Some((i32::from_be_bytes(slice.try_into().ok()?), offset + 4))
}

/// The six counts a TZif header carries, in file order (RFC 8536 §3.1):
/// `isutcnt`, `isstdcnt`, `leapcnt`, `timecnt`, `typecnt`, `charcnt`.
struct HeaderCounts {
    isutcnt: u32,
    isstdcnt: u32,
    leapcnt: u32,
    timecnt: u32,
    typecnt: u32,
    charcnt: u32,
}

/// Reads one 44-byte TZif header (magic + version + 15 reserved bytes
/// + six 4-byte counts) at `offset`. Returns the version byte, the
/// counts, and the offset of the data block that follows.
fn read_header(bytes: &[u8], offset: usize) -> Option<(u8, HeaderCounts, usize)> {
    let magic = bytes.get(offset..offset + 4)?;
    if magic != b"TZif" {
        return None;
    }
    let version = *bytes.get(offset + 4)?;
    // 4 (magic) + 1 (version) + 15 (reserved) = 20 bytes before the counts
    let mut pos = offset + 20;
    let (isutcnt, next) = read_u32(bytes, pos)?;
    pos = next;
    let (isstdcnt, next) = read_u32(bytes, pos)?;
    pos = next;
    let (leapcnt, next) = read_u32(bytes, pos)?;
    pos = next;
    let (timecnt, next) = read_u32(bytes, pos)?;
    pos = next;
    let (typecnt, next) = read_u32(bytes, pos)?;
    pos = next;
    let (charcnt, next) = read_u32(bytes, pos)?;
    pos = next;
    Some((
        version,
        HeaderCounts { isutcnt, isstdcnt, leapcnt, timecnt, typecnt, charcnt },
        pos,
    ))
}

/// Reads one data block — either the v1 (32-bit transition times) or
/// the v2/v3 (64-bit transition times) body that follows a header —
/// and returns the parsed `TzFile` plus the offset just past this
/// block (leap-second records, standard/wall indicators, and UT/local
/// indicators are skipped: `utc_offset_seconds_for_wall_time` needs
/// only the transition table and the type table).
fn read_data_block(bytes: &[u8], counts: &HeaderCounts, offset: usize, time_width: usize) -> Option<(TzFile, usize)> {
    let mut pos = offset;
    let mut transition_at = Vec::with_capacity(counts.timecnt as usize);
    for _ in 0..counts.timecnt {
        if time_width == 8 {
            let (value, next) = read_i64(bytes, pos)?;
            transition_at.push(value);
            pos = next;
        } else {
            let (value, next) = read_i32(bytes, pos)?;
            transition_at.push(value as i64);
            pos = next;
        }
    }
    let mut transition_type = Vec::with_capacity(counts.timecnt as usize);
    for _ in 0..counts.timecnt {
        let byte = *bytes.get(pos)?;
        transition_type.push(byte);
        pos += 1;
    }
    let mut types = Vec::with_capacity(counts.typecnt as usize);
    for _ in 0..counts.typecnt {
        let (utoff, next) = read_i32(bytes, pos)?;
        pos = next;
        let is_dst = *bytes.get(pos)? != 0;
        pos += 1;
        // desigidx (1 byte, the abbreviation string's start index) —
        // unread: `utc_offset_seconds_for_wall_time` needs only the
        // offset and the DST flag.
        pos += 1;
        types.push(LocalTimeType { utoff_seconds: utoff as i64, is_dst });
    }
    // abbreviation characters
    pos += counts.charcnt as usize;
    // leap-second records: each is a transition time (`time_width`
    // bytes) plus a 4-byte correction
    pos += counts.leapcnt as usize * (time_width + 4);
    // standard/wall indicators, then UT/local indicators
    pos += counts.isstdcnt as usize;
    pos += counts.isutcnt as usize;
    Some((TzFile { transition_at, transition_type, types }, pos))
}

/// Parses a whole TZif file's bytes. A v1-only file (`version == 0`,
/// the NUL byte RFC 8536 spells for "no v2 block") reads its one
/// 32-bit block; a v2/v3 file reads PAST the v1 block (present only
/// for pre-1970 tools, RFC 8536 §3.2) to the v2/v3 block, whose
/// 64-bit transition times are the ones this reader uses — the same
/// "prefer the 64-bit block when present" rule every TZif reader
/// follows, since the 32-bit block cannot represent instants outside
/// the `i32` range.
fn parse_tzif(bytes: &[u8]) -> Option<TzFile> {
    let (version, v1_counts, v1_data_offset) = read_header(bytes, 0)?;
    let (v1_file, after_v1) = read_data_block(bytes, &v1_counts, v1_data_offset, 4)?;
    if version == 0 {
        return Some(v1_file);
    }
    // v2/v3: a second header immediately follows the v1 data block,
    // then the 64-bit body.
    let (_version2, v2_counts, v2_data_offset) = read_header(bytes, after_v1)?;
    let (v2_file, _after_v2) = read_data_block(bytes, &v2_counts, v2_data_offset, 8)?;
    Some(v2_file)
}

/// The zoneinfo search roots, tried in order — `/usr/share/zoneinfo`
/// is where this system's tzdata lives (confirmed: `Europe/Paris`
/// there is TZif version 2, "fat" — both a 32-bit and a 64-bit block).
const ZONEINFO_ROOTS: [&str; 2] = ["/usr/share/zoneinfo", "/usr/lib/zoneinfo"];

/// Reads and parses the named zone's TZif file (`"Europe/Paris"` →
/// `<root>/Europe/Paris`). `None` when no root holds the file or it
/// fails to parse — the caller (`classify_tzinfo_expression`) treats
/// that exactly like today's `TzinfoKind::OtherAware`, never a guess.
fn load_zone(zone_name: &str) -> Option<TzFile> {
    // A zone name is a relative POSIX path (`Area/Location`, sometimes
    // `Area/Location/Sublocation`) — reject any component that would
    // escape the zoneinfo root (`..`, an absolute leading slash, an
    // empty component) rather than trust the source string.
    if zone_name.is_empty() || zone_name.starts_with('/') {
        return None;
    }
    if zone_name.split('/').any(|part| part.is_empty() || part == "." || part == "..") {
        return None;
    }
    for root in ZONEINFO_ROOTS {
        let mut path = PathBuf::from(root);
        path.push(zone_name);
        if let Ok(bytes) = fs::read(&path) {
            if let Some(file) = parse_tzif(&bytes) {
                return Some(file);
            }
        }
    }
    None
}

/// The index of the local-time-type in effect for the UTC instant
/// `epoch_seconds`, by RFC 8536 §3.2's own rule: the LAST transition
/// at or before the instant; if the instant is before every
/// transition (or the zone has none), the first type whose `is_dst`
/// is false, or type 0 if all types are DST, or type 0 if there are
/// no types at all (unreachable for a real zone file, but `types` is
/// read as data, not assumed non-empty).
fn type_index_for_instant(file: &TzFile, epoch_seconds: i64) -> Option<usize> {
    if file.types.is_empty() {
        return None;
    }
    let mut chosen: Option<usize> = None;
    for (i, &at) in file.transition_at.iter().enumerate() {
        if at <= epoch_seconds {
            chosen = Some(file.transition_type[i] as usize);
        } else {
            break;
        }
    }
    if let Some(index) = chosen {
        return Some(index);
    }
    let fallback = file.types.iter().position(|t| !t.is_dst).unwrap_or(0);
    Some(fallback)
}

/// Days since the epoch (1970-01-01) for a proleptic-Gregorian civil
/// date — Howard Hinnant's `days_from_civil` algorithm (`chrono::
/// NaiveDate`'s own internal algorithm; correct for every year this
/// crate's corpus constructs, including years before 1970). The one
/// calendar-arithmetic helper this reader needs to turn `datetime`
/// construction fields into an epoch-seconds value for tzdata lookup
/// — ordinary Gregorian day counting, not a refined-set question, so
/// it does not go through the kernel's calendar seam
/// (`calendar_interpreter.rs`'s own `epoch_days_of`, which answers a
/// DIFFERENT question: whether a date is IN RANGE, not its day count).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let mp = (month + 9) % 12; // [0, 11]
    let doy = (153 * mp + 2) / 5 + day - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Epoch seconds (as if the wall-clock fields were already UTC) for a
/// full calendar+time-of-day reading — the input `utc_offset_seconds_
/// for_wall_time` and its own fixed-point search take.
pub fn wall_clock_epoch_seconds(year: i64, month: i64, day: i64, hour: i64, minute: i64, second: i64) -> i64 {
    days_from_civil(year, month, day) * 86400 + hour * 3600 + minute * 60 + second
}

/// The UTC offset (seconds east of UTC) in effect for one LOCAL
/// (wall-clock, zone-naive) instant in the named zone, at the given
/// calendar fields. Converts local to UTC by the standard fixed-point
/// approach: guess the offset from the instant read as if it were
/// already UTC, convert wall time to a UTC epoch estimate with that
/// guess, look up the offset actually in effect at THAT UTC instant,
/// and accept once a second lookup confirms the same offset (the
/// ordinary, non-transitioning case — the only one this reader needs,
/// since `expressions.rs` calls it only for construction literals with
/// fixed calendar fields, never for a rule about ambiguous/imaginary
/// local times during a DST transition). `None` when the zone cannot
/// be loaded/parsed, or when the fixed point does not settle within a
/// few iterations (a transition-boundary wall time this reader
/// declines rather than guesses).
pub fn utc_offset_seconds_for_wall_time(zone_name: &str, epoch_seconds_as_utc: i64) -> Option<i64> {
    let file = load_zone(zone_name)?;
    let mut guess_offset = {
        let index = type_index_for_instant(&file, epoch_seconds_as_utc)?;
        file.types[index].utoff_seconds
    };
    // A handful of iterations settles any real zone's offset (offsets
    // never change more than once per lookup in practice); a zone that
    // never converges is declined rather than guessed.
    for _ in 0..4 {
        let candidate_utc = epoch_seconds_as_utc - guess_offset;
        let index = type_index_for_instant(&file, candidate_utc)?;
        let candidate_offset = file.types[index].utoff_seconds;
        if candidate_offset == guess_offset {
            return Some(candidate_offset);
        }
        guess_offset = candidate_offset;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn europe_paris_summer_2024_is_plus_two_hours() {
        // 2024-06-01 12:00:00 local, Europe/Paris — CEST (UTC+2),
        // summer time is in effect (DST ends the last Sunday of
        // October, starts the last Sunday of March).
        let wall = wall_clock_epoch_seconds(2024, 6, 1, 12, 0, 0);
        let offset = utc_offset_seconds_for_wall_time("Europe/Paris", wall);
        assert_eq!(offset, Some(7200));
    }

    #[test]
    fn europe_paris_winter_2024_is_plus_one_hour() {
        // 2024-01-15 12:00:00 local, Europe/Paris — CET (UTC+1),
        // standard time (outside the DST window).
        let wall = wall_clock_epoch_seconds(2024, 1, 15, 12, 0, 0);
        let offset = utc_offset_seconds_for_wall_time("Europe/Paris", wall);
        assert_eq!(offset, Some(3600));
    }

    #[test]
    fn an_unknown_zone_name_declines() {
        assert_eq!(utc_offset_seconds_for_wall_time("Not/AZone", 0), None);
    }

    #[test]
    fn a_zone_name_that_would_escape_the_root_declines() {
        assert_eq!(utc_offset_seconds_for_wall_time("../etc/passwd", 0), None);
        assert_eq!(utc_offset_seconds_for_wall_time("/etc/passwd", 0), None);
    }
}
