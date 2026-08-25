//! The manifest reader's own decline vocabulary — an unreadable
//! manifest file, a module named but with no matching entry, a crossing
//! argument that escapes an entry's declared sort, a return crossing
//! with no producer half yet — and the `datetime.strptime` STAGE 2
//! per-directive declines (unread vs. locale-dependent).

/// The manifest reader's own DECLINE sentence for a module that IS named
/// in a manifest but the CALLED function is not one of the manifest's
/// listed entries — a narrower named decline than the bare unmodeled-
/// module sentence, since the manifest at least states what it does
/// cover.
pub fn manifest_names_no_entry_for(module_name: &str, function_name: &str) -> String {
    format!(
        "'{module_name}''s manifest names no entry for '{function_name}' — the call is a manifested module's \
        own function this checker still has no contract for"
    )
}

/// `datetime.strptime(text, format)` date.12 STAGE 2's own named decline
/// for a format string naming a directive this round has not
/// transcribed against datetime.rst's format-codes table yet (`%z %Z
/// %I %G %u %V` — `expressions.rs::Strptime2Decline::UnreadDirective`'s
/// own set). Names the ONE directive that blocked the read, never the
/// whole format string — a host-independent value set is buildable for
/// this directive once transcribed; today it simply is not yet.
pub fn strptime_unread_directive(letter: char) -> String {
    format!(
        "this format string names the directive '%{letter}', which this checker has not yet transcribed \
        against datetime.rst's format-codes table"
    )
}

/// `datetime.strptime(text, format)` date.12 STAGE 2's own named decline
/// for a format string naming a LOCALE-dependent directive (`%a %A %b
/// %B %p %c %x %X` — `expressions.rs::Strptime2Decline::LocaleDirective`'s
/// own set) — datetime.rst note (1): "the format depends on the current
/// locale... Field orderings will vary... and the output may contain
/// non-ASCII characters." A genuinely distinct reason from
/// `strptime_unread_directive`'s: no host-independent value set exists
/// for a locale directive AT ALL, not merely one this round left
/// untranscribed.
pub fn strptime_locale_directive(letter: char) -> String {
    format!(
        "this format string names the directive '%{letter}', which reads a value from the host's locale \
        (datetime.rst note 1) — there is no host-independent set for a locale-dependent directive to derive"
    )
}

/// The manifest reader's own DECLINE sentence for a manifest file this
/// reader could not parse at all — the whole manifest is unusable, so
/// every call into the module it would have covered stays the bare
/// unmodeled-module decline instead.
pub fn manifest_unreadable(manifest_path: &str, reason: &str) -> String {
    format!("the manifest {manifest_path} could not be read: {reason}")
}

/// A crossing argument escapes the manifest entry's own declared sort —
/// the manifest lane's own crossing-fit refusal, the same shape the
/// stdio edge's `containment_refutation` fires, restated for a
/// manifest-declared parameter (a plain sort word, never a full
/// `DeclaredRefinement` spelling).
pub fn manifest_entry_crossing_refused(
    module_name: &str,
    function_name: &str,
    parameter_name: &str,
    value_words: &str,
    declared_sort: &str,
) -> String {
    format!(
        "a value of type '{value_words}' is not assignable to '{module_name}.{function_name}''s declared \
        parameter '{parameter_name}: {declared_sort}' — the manifest states the entry contract, and \
        {value_words} escapes it"
    )
}

/// The manifest reader's own DECLINE sentence for a call whose return
/// crosses the entry contract fit but has no producer half yet — the
/// manifest states the ENTRY, never the return; a later unit (the
/// producer half, python-c-extension-boundary.md build order item 3)
/// closes this. Names both the module/function and the missing producer
/// symbol so the decline reads as a work-queue item.
pub fn manifest_entry_names_no_producer(module_name: &str, function_name: &str, producer_symbol: &str) -> String {
    format!(
        "'{module_name}.{function_name}''s manifest names its entry but no producer exports its return fact \
        (the manifest names the producer symbol '{producer_symbol}', and no C++/native adapter has exported a \
        fact for it yet)"
    )
}
