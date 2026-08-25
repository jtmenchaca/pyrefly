//! The Match-object value family: `re.fullmatch`/`re.finditer`'s own
//! yielded match answer (`match_object_value`) and `match.group(n)`'s
//! own reader (`matched_group_grammar`), plus the private capture-group
//! parser (`capture_group_spans`) both routes need.

use refined_domain::abstract_value::{known_set, AbstractValue, Kind, ObjectKey, PrimitiveKind, SetKindTag};
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;
use refined_sets::regex_compiler::format_grammar;

/// The word a Match-object value built by `match_object_value` carries
/// on `kind_word` — distinct from `evaluate_attribute_call`'s existing
/// bare `"a match object"` opaque tag (`re.match`/`re.search`'s own
/// contentless answer), since THIS value additionally carries readable
/// group grammars a caller's `.group(n)` needs to find. `expressions.rs`
/// reads this word to route a `.group(...)` call through
/// `matched_group_grammar` below rather than the opaque-value default.
pub const MATCH_WITH_GROUPS_WORD: &str = "a match object with readable groups";

/// `pattern` with `^` and `$` present at its two ends, adding whichever
/// it lacks. A trailing `\$` is an escaped dollar CHARACTER, not an end
/// anchor, so it still gets its own `$` appended.
fn anchored(pattern: &str) -> String {
    let mut text = pattern.to_owned();
    if !text.starts_with('^') {
        text.insert(0, '^');
    }
    if !(text.ends_with('$') && !text.ends_with("\\$")) {
        text.push('$');
    }
    text
}

/// One top-level capturing group of a regex pattern: the group's own
/// inner text (no enclosing parens) and, for a `(?P<name>...)` group,
/// the name it is additionally reachable by.
pub(super) struct CaptureGroup {
    pub body: String,
    pub name: Option<String>,
}

/// The top-level PARENTHESIZED groups of a regex pattern, in
/// left-to-right opening order — `re.fullmatch(pattern, s)`'s own
/// capture-group numbering (library/re.html, "Group 0 is the entire
/// match... groups are numbered from 1 in the order their opening
/// parentheses appear").
///
/// Recognizes only the shapes the corpus's own patterns and
/// `format_grammar`'s own supported subset both need: plain capturing
/// groups `(...)`, NAMED capturing groups `(?P<name>...)`
/// (library/re.html, "(?P<name>...) Similar to regular parentheses, but
/// the substring matched by the group is accessible via the symbolic
/// group name *name*" — a named group is numbered exactly like a plain
/// one AND reachable by its name), with `\(`/`\)` escapes and NESTED
/// parens read as plain text ONE LEVEL DEEP (a nested group inside a
/// captured group is not itself extracted as a separate numbered group
/// — this reader finds no corpus row needing that). A non-capturing
/// group `(?:...)` is recognized and its own parens are consumed but it
/// contributes NO numbered group — `re.rst`'s own "the contents of a
/// group ... `(?:...)` ... cannot be retrieved". An unmatched paren, or
/// a `(?...)` extension other than `(?:` and `(?P<`, makes the WHOLE
/// read decline (`None`) — this is not a general regex parser, only
/// enough to find each `(...)`'s own span in the corpus's own literal
/// patterns.
pub(super) fn capture_group_spans(pattern: &str) -> Option<Vec<CaptureGroup>> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut groups = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                i += 2; // an escaped character is never a group boundary
            }
            '(' => {
                let is_non_capturing = chars.get(i + 1) == Some(&'?') && chars.get(i + 2) == Some(&':');
                // a `(?P<name>` opening: read the name and start the
                // body after the closing `>`.
                let named = if chars.get(i + 1) == Some(&'?') && chars.get(i + 2) == Some(&'P') && chars.get(i + 3) == Some(&'<') {
                    let name_start = i + 4;
                    let name_end = (name_start..chars.len()).find(|&k| chars[k] == '>')?;
                    Some((chars[name_start..name_end].iter().collect::<String>(), name_end + 1))
                } else {
                    None
                };
                if chars.get(i + 1) == Some(&'?') && !is_non_capturing && named.is_none() {
                    return None; // an unsupported (?...) extension
                }
                let body_start = match (&named, is_non_capturing) {
                    (Some((_, after_name)), _) => *after_name,
                    (None, true) => i + 3,
                    (None, false) => i + 1,
                };
                let mut depth = 1;
                let mut j = body_start;
                while j < chars.len() && depth > 0 {
                    match chars[j] {
                        '\\' => j += 1,
                        '(' => depth += 1,
                        ')' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                if depth != 0 {
                    return None; // an unmatched opening paren
                }
                let body_end = j - 1;
                if !is_non_capturing {
                    groups.push(CaptureGroup {
                        body: chars[body_start..body_end].iter().collect(),
                        name: named.map(|(name, _)| name),
                    });
                }
                i = j;
            }
            _ => i += 1,
        }
    }
    Some(groups)
}

/// The Match-object value `re.fullmatch(pattern, subject)` /
/// `re.finditer(pattern, subject)`'s own yielded match answer —
/// library/re.html: `fullmatch(pattern, string)` "If the whole string
/// matches this regular expression, return a corresponding match
/// object"; `finditer(pattern, string)` "Return an iterator yielding
/// match objects." A `Kind::Object` (`MATCH_WITH_GROUPS_WORD`-tagged)
/// whose keys are `"0"` (the whole match, ALWAYS present — "group()"
/// with no argument or `group(0)` "The entire match") through `"N"`
/// (each capturing group, in `capture_group_spans`'s own left-to-right
/// numbering), every key's value the group's OWN compiled grammar,
/// ANCHORED on both ends.
///
/// Every key — group `"0"` included — is the text the match ITSELF
/// spans, never that text embedded in surrounding context:
/// library/re.html, "Match.group([group, ...]) ... Returns one or more
/// subgroups of the match", and group 0 "The entire match" is the
/// substring the pattern consumed, whose first and last code points are
/// the match's own boundaries. `format_grammar` pads an unanchored side
/// with `C*` (a regex matches a SUBSTRING of its subject) — correct
/// when compiling a pattern to ask what SUBJECTS it accepts, and wrong
/// here, where the value being described is the matched substring
/// itself. So each grammar is compiled with `^`/`$` inserted, and the
/// caller's own anchoring is irrelevant to what a group READS BACK: a
/// `finditer` iteration over `[A-Z]{2}` yields matches whose `group(0)`
/// is exactly two upper-case letters, not two letters surrounded by
/// arbitrary text.
///
/// `None` on a pattern `capture_group_spans` cannot read, or a compiled
/// grammar `format_grammar` refuses for group 0 or any numbered group —
/// the WHOLE match value declines rather than answer a partial object
/// missing some groups.
pub fn match_object_value(pattern: &str) -> Option<AbstractValue> {
    let whole_grammar = format_grammar(&anchored(pattern), "");
    if !whole_grammar.ok {
        return None;
    }
    let groups = capture_group_spans(pattern)?;
    let mut keys = vec![ObjectKey {
        name: "0".to_owned(),
        numeric: true,
        value: AbstractValue {
            kind_tag: Some(PrimitiveKind::String),
            ..known_set(whole_grammar.set, None, TrustProved, SetKindTag::None)
        },
    }];
    for (index, group) in groups.iter().enumerate() {
        let compiled = format_grammar(&anchored(&group.body), "");
        if !compiled.ok {
            return None;
        }
        let value = AbstractValue {
            kind_tag: Some(PrimitiveKind::String),
            ..known_set(compiled.set, None, TrustProved, SetKindTag::None)
        };
        keys.push(ObjectKey {
            name: (index + 1).to_string(),
            numeric: true,
            value: value.clone(),
        });
        // a `(?P<name>...)` group is reachable BOTH by its number and by
        // its name (library/re.html, "(?P<name>...)"), so it carries a
        // second, non-numeric key holding the identical grammar.
        if let Some(name) = &group.name {
            keys.push(ObjectKey {
                name: name.clone(),
                numeric: false,
                value,
            });
        }
    }
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.kind_word = Some(MATCH_WITH_GROUPS_WORD);
    Some(instance)
}

/// `match.group(n)` / `match.group("name")` (one-argument form) over a
/// `match_object_value`-built receiver — library/re.html#re.Match.group:
/// "If a single argument is used, result is a single string." Group `0`
/// (or the no-argument default, `group()`'s own zero-arg row — not this
/// function, which only ever reads the one-argument form) is the whole
/// match; group `N` (`N >= 1`) is that numbered capturing group's own
/// compiled grammar; a STRING argument names a `(?P<name>...)` group
/// ("the substring matched by the group is accessible via the symbolic
/// group name *name*"), reading the name key `match_object_value` built
/// beside that group's number key. A group this match's own value
/// carries no key for (a number out of range, or a name the pattern
/// never declares) declines — CPython raises `IndexError: no such
/// group` for it, a fact this row does not itself speak the raise for
/// (no exception channel in this file); a non-Match receiver, or an
/// argument that is neither a known Integer nor a known string,
/// declines the same way.
pub fn matched_group_grammar(receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if receiver.kind != Kind::Object || receiver.kind_word != Some(MATCH_WITH_GROUPS_WORD) {
        return None;
    }
    let [selector] = arguments else { return None };
    if selector.kind != Kind::Values {
        return None;
    }
    let (name, numeric) = match selector.kind_tag {
        Some(PrimitiveKind::Integer) if selector.values.len() == 1 => (format!("{}", selector.values[0] as i64), true),
        Some(PrimitiveKind::String) => (super::exact_string_text(selector)?, false),
        _ => return None,
    };
    receiver
        .keys
        .iter()
        .find(|key| key.numeric == numeric && key.name == name)
        .map(|key| key.value.clone())
}
