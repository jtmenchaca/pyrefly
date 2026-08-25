//! What names one `case` pattern binds, syntactically
//! (`pattern_captures`) and with the value each name PROVABLY holds
//! against a known subject (`pattern_bound_captures`).

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::Pattern;

use crate::collection_models::subscript_read;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::instances::field_read;
use crate::instances::ClassModel;

use super::value_proof::pattern_proved_value;

/// The field names a `MatchClass` pattern's own class name resolves to,
/// in `__match_args__`/declaration order — `None` when `classes` carries
/// no table, the pattern's `cls` expression is not a bare Name, or the
/// name is not in the table (an imported/builtin class this checker's
/// class table never populates, e.g. `case int():`). Shared by
/// `pattern_captures` and `pattern_bound_captures` so a positional
/// class pattern's field-order lookup is written once.
fn class_pattern_fields<'a>(
    class_pattern: &ruff_python_ast::PatternMatchClass,
    classes: Option<&'a HashMap<String, ClassModel>>,
) -> Option<&'a [crate::instances::ClassField]> {
    let Expr::Name(class_name) = class_pattern.cls.as_ref() else {
        return None;
    };
    let classes = classes?;
    let model = classes.get(class_name.id.as_str())?;
    Some(&model.fields)
}

/// Whether a `MatchMapping` key expression is a literal this file can
/// read as a fixed key spelling — a string literal, the only key shape
/// this corpus's mapping-pattern rows use (`case {"age": bound_age}:`).
/// Any other expression shape (a dotted constant, a number, an
/// f-string) answers `false` — not read this wave.
fn is_literal_mapping_key(key: &Expr) -> bool {
    matches!(key, Expr::StringLiteral(_))
}

/// The bare names one `case` pattern captures — a SYNTACTIC question,
/// answered without deciding whether the pattern would actually take
/// (that question is `pattern_outcome`'s, not this function's).
/// `Pattern::MatchValue`/`MatchSingleton` bind nothing. `Pattern::MatchAs`
/// binds its own `name` (a bare capture/wildcard has no inner pattern)
/// plus whatever its inner pattern (if any) itself binds.
/// `Pattern::MatchOr` recurses into its FIRST alternative only — Python's
/// own grammar rule (compound_stmts.rst, "the same set of names must be
/// captured by all the alternatives") makes every alternative's own
/// capture set identical, so any one alternative names the whole
/// pattern's captures.
///
/// `Pattern::MatchSequence` names every bare-Name capture in its
/// `patterns` list positionally, plus a `MatchStar` element's own name
/// (`case [first, *rest]:` names both `first` and `rest`; a wildcard
/// star `*_` names nothing, matching PEP 634's "`_` never binds"). Any
/// element that is not itself a bare-Name/wildcard `MatchAs` (a nested
/// literal, sequence, or class sub-pattern) makes the WHOLE sequence
/// pattern decline — this function reads only the flat bare-capture
/// case, never recurses past one level into a structural sub-pattern.
///
/// `Pattern::MatchMapping` names every value-side capture whose KEY is
/// a literal (a `MatchValue`/`MatchSingleton`-free syntactic literal —
/// in practice a string, this corpus's only mapping-key shape) and
/// whose value-side pattern is itself a bare-Name/wildcard `MatchAs`,
/// plus the `**rest` capture (`rest: Option<Identifier>`) when present.
/// A non-literal key, or a value-side pattern that is not a bare
/// capture, declines the whole mapping pattern.
///
/// `Pattern::MatchClass` binds nothing ITSELF (a bare `case int():`
/// names nothing). It is nameable in two shapes: NO sub-patterns at
/// all (`case int() as n:`, `arguments.patterns`/`.keywords` both
/// empty), or KEYWORD sub-patterns ONLY, each itself a bare-Name/
/// wildcard `MatchAs` (`case Point(x=px):` names `px`) — a keyword's
/// own `attr` IS the field name, so naming it needs no class lookup.
/// POSITIONAL sub-patterns (`case Point(px, py):`) resolve through the
/// class's own `__match_args__` order (`ClassModel.fields`, pydantic's
/// own declaration-order convention, `class_pattern_fields`'s own doc):
/// each positional bare-Name/wildcard sub-pattern names the field at its
/// own position. A pattern with MORE positions than the class has
/// fields declines whole (Python itself raises `TypeError` for this
/// shape at runtime; this function never guesses a truncated binding).
/// `classes` is `None` when no caller has a class table to offer (this
/// function's own tests, and any future caller outside a match walk) —
/// every positional pattern then declines exactly as before this
/// capability existed.
pub fn pattern_captures(pattern: &Pattern, classes: Option<&HashMap<String, ClassModel>>) -> Option<Vec<String>> {
    match pattern {
        Pattern::MatchValue(_) | Pattern::MatchSingleton(_) => Some(Vec::new()),
        Pattern::MatchAs(as_pattern) => {
            let mut names = match as_pattern.pattern.as_deref() {
                Some(inner) => pattern_captures(inner, classes)?,
                None => Vec::new(),
            };
            if let Some(name) = as_pattern.name.as_ref() {
                names.push(name.id.as_str().to_owned());
            }
            Some(names)
        }
        Pattern::MatchOr(or_pattern) => {
            let first = or_pattern.patterns.first()?;
            pattern_captures(first, classes)
        }
        Pattern::MatchSequence(sequence_pattern) => {
            let mut names = Vec::new();
            for element in &sequence_pattern.patterns {
                match element {
                    Pattern::MatchStar(star) => {
                        if let Some(name) = star.name.as_ref() {
                            names.push(name.id.as_str().to_owned());
                        }
                    }
                    Pattern::MatchAs(as_pattern) if as_pattern.pattern.is_none() => {
                        if let Some(name) = as_pattern.name.as_ref() {
                            names.push(name.id.as_str().to_owned());
                        }
                    }
                    // a nested literal/sequence/mapping/class sub-pattern
                    // — beyond this function's flat bare-capture scope
                    _ => return None,
                }
            }
            Some(names)
        }
        Pattern::MatchMapping(mapping_pattern) => {
            if mapping_pattern.keys.len() != mapping_pattern.patterns.len() {
                return None;
            }
            let mut names = Vec::new();
            for (key, value_pattern) in mapping_pattern.keys.iter().zip(mapping_pattern.patterns.iter()) {
                if !is_literal_mapping_key(key) {
                    return None;
                }
                let Pattern::MatchAs(as_pattern) = value_pattern else {
                    return None;
                };
                if as_pattern.pattern.is_some() {
                    return None;
                }
                if let Some(name) = as_pattern.name.as_ref() {
                    names.push(name.id.as_str().to_owned());
                }
            }
            if let Some(rest) = mapping_pattern.rest.as_ref() {
                names.push(rest.id.as_str().to_owned());
            }
            Some(names)
        }
        Pattern::MatchClass(class_pattern) => {
            let mut names = Vec::new();
            if !class_pattern.arguments.patterns.is_empty() {
                let fields = class_pattern_fields(class_pattern, classes)?;
                if class_pattern.arguments.patterns.len() > fields.len() {
                    // more positions than the class declares fields —
                    // Python itself raises TypeError for this shape
                    return None;
                }
                for sub_pattern in class_pattern.arguments.patterns.iter() {
                    let Pattern::MatchAs(as_pattern) = sub_pattern else {
                        return None;
                    };
                    if as_pattern.pattern.is_some() {
                        return None;
                    }
                    if let Some(name) = as_pattern.name.as_ref() {
                        names.push(name.id.as_str().to_owned());
                    }
                }
            }
            for keyword in &class_pattern.arguments.keywords {
                let Pattern::MatchAs(as_pattern) = &keyword.pattern else {
                    return None;
                };
                if as_pattern.pattern.is_some() {
                    return None;
                }
                if let Some(name) = as_pattern.name.as_ref() {
                    names.push(name.id.as_str().to_owned());
                }
            }
            Some(names)
        }
        Pattern::MatchStar(_) => None,
    }
}

/// The (name, value) pair every capture `pattern_captures` names,
/// filled in with the value each name PROVABLY holds when `subject` is
/// known — the value-bearing counterpart naming alone cannot answer.
/// `None` means `pattern` itself has no nameable captures — this
/// function decides `Some`/`None` on the SAME conditions
/// `pattern_captures` does (a caller needing only the names, never the
/// values, still has that lighter-weight function to call; `check.rs::
/// walk_match`'s join-fallback path calls THIS one directly, since it
/// always needs both in the same pass).
///
/// A `MatchAs`'s own captured name binds to `pattern_proved_value`'s
/// proof for the pattern rooted at that `as` (e.g. `(40 | 41) as
/// chosen` binds `chosen` to `{40, 41}`, not the raw subject) when one
/// exists, falling back to `subject` itself for a bare capture/wildcard
/// (`pattern_proved_value` proves nothing for those, by design — the
/// SAME fallback `check.rs::walk_match` already applies for a
/// literal/singleton/or/as pattern with no sequence/mapping/class
/// shape involved).
///
/// A capture whose OWN value cannot be proved from `subject` (an
/// unknown/wrong-kind receiver, an absent key, an out-of-range
/// position) binds `unknown()` for that one name rather than dropping
/// it or guessing — `assignability::judge`'s own law never fires an
/// `Unknown` value (only `Object`/`List`/`Null` structural mismatches
/// fire against a scalar declared set), so an unproved capture is
/// SILENT-SAFE: it reaches the sink Undetermined, never a false Fire.
pub fn pattern_bound_captures(
    pattern: &Pattern,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<(String, AbstractValue)>> {
    match pattern {
        Pattern::MatchValue(_) | Pattern::MatchSingleton(_) => Some(Vec::new()),
        Pattern::MatchAs(as_pattern) => {
            let mut bound = match as_pattern.pattern.as_deref() {
                Some(inner) => pattern_bound_captures(inner, subject, environment, kernel)?,
                None => Vec::new(),
            };
            if let Some(name) = as_pattern.name.as_ref() {
                let proved = pattern_proved_value(pattern, environment, kernel);
                let own_value = proved.unwrap_or_else(|| subject.clone());
                bound.push((name.id.as_str().to_owned(), own_value));
            }
            Some(bound)
        }
        Pattern::MatchOr(or_pattern) => {
            let first = or_pattern.patterns.first()?;
            pattern_bound_captures(first, subject, environment, kernel)
        }
        Pattern::MatchSequence(sequence_pattern) => {
            let items = if subject.kind == Kind::List { Some(&subject.items) } else { None };
            let mut bound = Vec::new();
            for (position, element) in sequence_pattern.patterns.iter().enumerate() {
                match element {
                    Pattern::MatchStar(star) => {
                        if let Some(name) = star.name.as_ref() {
                            // the remainder is a LIST, never a scalar this
                            // corpus's rows read at a refined sink — bound
                            // opaque rather than sliced out of `items`
                            bound.push((name.id.as_str().to_owned(), unknown()));
                        }
                    }
                    Pattern::MatchAs(as_pattern) if as_pattern.pattern.is_none() => {
                        if let Some(name) = as_pattern.name.as_ref() {
                            let element_value = items
                                .and_then(|items| items.get(position))
                                .cloned()
                                .unwrap_or_else(unknown);
                            bound.push((name.id.as_str().to_owned(), element_value));
                        }
                    }
                    _ => return None,
                }
            }
            Some(bound)
        }
        Pattern::MatchMapping(mapping_pattern) => {
            if mapping_pattern.keys.len() != mapping_pattern.patterns.len() {
                return None;
            }
            let mut bound = Vec::new();
            for (key, value_pattern) in mapping_pattern.keys.iter().zip(mapping_pattern.patterns.iter()) {
                if !is_literal_mapping_key(key) {
                    return None;
                }
                let Pattern::MatchAs(as_pattern) = value_pattern else {
                    return None;
                };
                if as_pattern.pattern.is_some() {
                    return None;
                }
                if let Some(name) = as_pattern.name.as_ref() {
                    let key_value = evaluate_expression(key, environment, kernel);
                    let bound_value = subscript_read(subject, &key_value).unwrap_or_else(unknown);
                    bound.push((name.id.as_str().to_owned(), bound_value));
                }
            }
            if let Some(rest) = mapping_pattern.rest.as_ref() {
                // `**rest` collects the remaining keys into a DICT, never
                // a scalar this corpus's rows read at a refined sink
                bound.push((rest.id.as_str().to_owned(), unknown()));
            }
            Some(bound)
        }
        Pattern::MatchClass(class_pattern) => {
            let mut bound = Vec::new();
            if !class_pattern.arguments.patterns.is_empty() {
                let classes = environment.classes().map(|classes| classes.as_ref());
                let fields = class_pattern_fields(class_pattern, classes)?;
                if class_pattern.arguments.patterns.len() > fields.len() {
                    // more positions than the class declares fields —
                    // Python itself raises TypeError for this shape
                    return None;
                }
                for (field, sub_pattern) in fields.iter().zip(class_pattern.arguments.patterns.iter()) {
                    let Pattern::MatchAs(as_pattern) = sub_pattern else {
                        return None;
                    };
                    if as_pattern.pattern.is_some() {
                        return None;
                    }
                    if let Some(name) = as_pattern.name.as_ref() {
                        let field_value = field_read(subject, field.name.as_str()).unwrap_or_else(unknown);
                        bound.push((name.id.as_str().to_owned(), field_value));
                    }
                }
            }
            for keyword in &class_pattern.arguments.keywords {
                let Pattern::MatchAs(as_pattern) = &keyword.pattern else {
                    return None;
                };
                if as_pattern.pattern.is_some() {
                    return None;
                }
                if let Some(name) = as_pattern.name.as_ref() {
                    let field_value = field_read(subject, keyword.attr.id.as_str()).unwrap_or_else(unknown);
                    bound.push((name.id.as_str().to_owned(), field_value));
                }
            }
            Some(bound)
        }
        Pattern::MatchStar(_) => None,
    }
}
