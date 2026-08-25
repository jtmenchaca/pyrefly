//! Ascii-case conjunction, `re` module calls, and `all()` generator predicates.

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::regex_compiler::format_grammar;
use ruff_python_ast::Expr;

use crate::env::Environment;

use super::bool_op::narrow;
use super::bool_op::narrow_set_kind_names;
use super::condition_tree::meet_set_answer;
use super::name_of;

/// `x.isascii() and x.isupper()` / `x.isascii() and x.islower()`, found
/// TOGETHER anywhere among `operands` (an `and` chain's own flat operand
/// list — `len(x) == 2 and x.isascii() and x.isupper()` is one
/// three-value `BoolOp`, F2.fixed's own shape), narrows `x`'s codepoint
/// ALPHABET to exactly the ASCII cased-letter window: `[0x41, 0x5A]` for
/// `isupper()`, `[0x61, 0x7A]` for `islower()`.
///
/// Neither call alone states this bound: `str.isascii()` alone only
/// proves every code point sits in `[0x00, 0x7F]` (stdtypes.rst,
/// `str.isascii()` — "ASCII characters have code points in the range
/// U+0000-U+007F"), and `str.isupper()`/`str.islower()` alone are pinned
/// only against the full Unicode "cased character" categories
/// (stdtypes.rst's own `[4]` footnote), which include cased letters far
/// outside ASCII (e.g. 'É', 'ß') — bounding either call by itself to
/// `[0x41,0x5A]`/`[0x61,0x7A]` would overclaim. Restricted to ASCII BY
/// `isascii()` in the same conjunction, though, the codepoints
/// `isupper()`/`islower()` can additionally hold narrow to EXACTLY the
/// ASCII cased letters: within `[0x00, 0x7F]`, the only cased code
/// points at all are `A`-`Z` (`0x41`-`0x5A`) and `a`-`z` (`0x61`-`0x7A`)
/// — every other ASCII code point (control characters, digits,
/// punctuation, space) is uncased, so "every cased character is
/// uppercase, and there is at least one" restricted to that alphabet
/// collapses to "every code point is in `[0x41, 0x5A]`."
///
/// Reads and rebuilds through `as_repetition`/`repeat_of`, the same
/// element-preserving pattern `narrow_name_length_against_literal` uses
/// for the LENGTH half of this same guard — this leaf tightens the
/// ELEMENT instead, so the two compose regardless of which operand the
/// source lists first (each leaf reads whatever the OTHER already
/// narrowed, since both run against the same shared `environment`).
/// Only the TRUE arm narrows: "not ASCII" or "not uppercase" states no
/// single alphabet this window can name (the excluded codepoints are
/// scattered, not a window), matching `narrow_regex_module_call`'s own
/// "no complement" default for a state this grammar cannot express.
/// Every other shape (no `isascii()` call, no `isupper()`/`islower()`
/// call, receivers naming different places, a non-Set binding) narrows
/// nothing — the honest default every leaf in this file keeps.
pub(super) fn narrow_ascii_case_conjunction(operands: &[Expr], environment: &mut Environment, truth: bool) {
    if !truth {
        return;
    }
    let Some(name) = operands.iter().find_map(is_isascii_call_name) else {
        return;
    };
    let Some((case_name, ascii_case)) = operands.iter().find_map(is_ascii_case_call) else {
        return;
    };
    if case_name != name {
        return;
    }
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Set {
        return;
    }
    let Some(repeated) = refined_sets::repetition_window_forms::as_repetition(&current.set) else {
        return;
    };
    let (lo, hi) = ascii_case.codepoint_window();
    let element = make_refined_set(vec![integer(), at_least(lo), at_most(hi)]);
    let grade = trust_level_of(&current);
    let narrowed_set = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(element, repeated.lo, repeated.hi)]);
    environment.bind(
        name,
        AbstractValue {
            kind_tag: current.kind_tag,
            ..known_set(narrowed_set, None, grade, current.set_kind_tag)
        },
    );
}

/// `A`-`Z` or `a`-`z` — the two ASCII cased-letter windows
/// `narrow_ascii_case_conjunction` narrows to, told apart by which call
/// (`isupper`/`islower`) named them.
#[derive(Clone, Copy)]
pub(super) enum AsciiCase {
    Upper,
    Lower,
}

impl AsciiCase {
    fn codepoint_window(self) -> (f64, f64) {
        match self {
            AsciiCase::Upper => (0x41 as f64, 0x5A as f64),
            AsciiCase::Lower => (0x61 as f64, 0x7A as f64),
        }
    }
}

/// Whether `expression` is `<bare name>.isascii()` — zero arguments, no
/// keywords, the receiver a bare tracked name. The tested place's own
/// name, or `None` for any other shape.
pub(super) fn is_isascii_call_name(expression: &Expr) -> Option<&str> {
    is_bare_string_predicate_call(expression, "isascii")
}

/// Whether `expression` is `<bare name>.isupper()` / `<bare name>.
/// islower()` — the tested place's own name paired with which case the
/// call names, or `None` for any other shape.
pub(super) fn is_ascii_case_call(expression: &Expr) -> Option<(&str, AsciiCase)> {
    if let Some(name) = is_bare_string_predicate_call(expression, "isupper") {
        return Some((name, AsciiCase::Upper));
    }
    if let Some(name) = is_bare_string_predicate_call(expression, "islower") {
        return Some((name, AsciiCase::Lower));
    }
    None
}

/// `<bare name>.<method>()` with zero arguments and no keywords — the
/// shape every `str` no-argument predicate call in this file reads,
/// shared by `isascii`/`isupper`/`islower` rather than duplicated three
/// times.
fn is_bare_string_predicate_call<'a>(expression: &'a Expr, method: &str) -> Option<&'a str> {
    let Expr::Call(call) = expression else { return None };
    let Expr::Attribute(attribute) = call.func.as_ref() else { return None };
    if attribute.attr.as_str() != method {
        return None;
    }
    if !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return None;
    }
    name_of(&attribute.value)
}


/// `re.fullmatch(pattern, name)` / `re.match` / `re.search` as the
/// whole condition: a truthy match object proves `name`'s string is in
/// the pattern's own language (library/re.html: `fullmatch` — "the
/// whole string matches"; `match` — "at the beginning of the string";
/// `search` — "the first location where"). The pattern compiles through
/// the SAME `format_grammar` the pydantic `pattern=` kwarg uses
/// (surface.rs), anchored to each function's own semantics: `fullmatch`
/// pins both ends, `match` the start, `search` neither
/// (`format_grammar` itself pads an unanchored side with C*). The
/// narrowed binding meets the compiled set into the current one,
/// dropping the bare C* string ground first — the kernel's aligned-
/// segment pattern prover reads one chain, never a stack (surface.rs's
/// own `pattern` branch documents the identical strip). The FALSE arm
/// narrows nothing: "no match" has no complement this grammar states.
/// A non-literal pattern, keyword or flag arguments, a non-name
/// subject, a non-Set binding, or a pattern `format_grammar` refuses
/// all decline — the honest default.
pub(super) fn narrow_regex_module_call(call: &ruff_python_ast::ExprCall, environment: &mut Environment, truth: bool) {
    if !truth {
        return;
    }
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return;
    };
    if name_of(attribute.value.as_ref()) != Some("re") {
        return;
    }
    let (anchor_start, anchor_end) = match attribute.attr.as_str() {
        "fullmatch" => (true, true),
        "match" => (true, false),
        "search" => (false, false),
        _ => return,
    };
    if !call.arguments.keywords.is_empty() || call.arguments.args.len() != 2 {
        return;
    }
    let Expr::StringLiteral(literal) = &call.arguments.args[0] else {
        return;
    };
    let Some(name) = name_of(&call.arguments.args[1]) else {
        return;
    };
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Set {
        return;
    }
    let mut pattern = literal.value.to_str().to_owned();
    if anchor_start && !pattern.starts_with('^') {
        pattern.insert(0, '^');
    }
    if anchor_end && !(pattern.ends_with('$') && !pattern.ends_with("\\$")) {
        pattern.push('$');
    }
    let mut grammar = format_grammar(&pattern, "");
    if !grammar.ok {
        return;
    }
    let ground = strings();
    let plain_ground = &ground.forms[0];
    let mut combined: Vec<_> = current.set.forms.iter().filter(|form| *form != plain_ground).cloned().collect();
    combined.extend(std::mem::take(&mut grammar.set.forms));
    let narrowed = AbstractValue {
        kind_tag: current.kind_tag,
        ..known_set(make_refined_set(combined), None, trust_level_of(&current), current.set_kind_tag)
    };
    environment.bind(name, narrowed);
}

/// `all(<predicate> for <var> in <name>)` proving TRUE (functions.rst,
/// `all(iterable)` — "Return True if all elements of the iterable are
/// true (or if the iterable is empty)"): every element `name` could ever
/// read satisfies `predicate`, so `name`'s own ELEMENT window narrows to
/// whatever `predicate` proves about a single drawn element — the
/// element-window channel `narrow_ascii_case_conjunction` already gives
/// a string's codepoint alphabet, generalized here to a `list`/`set`
/// receiver's own numeric element. Two receiver shapes:
///
/// - `Kind::Set` over a repetition window (`as_repetition` — a `list[X]`
///   parameter's own star seed, or an already-narrowed repetition):
///   every position draws from the SAME element set, so one sandbox ask
///   narrows the whole window's `element` at once, rebuilt through
///   `repeat_of` exactly as `narrow_ascii_case_conjunction`'s own
///   element tightening does.
/// - `Kind::List` (a fixed-arity literal, `lst = [a, b, c]`): there is
///   no single shared element — each SLOT holds its own binding, so
///   this asks the predicate once per item and meets the answer onto
///   that item alone (`meet_set_answer`'s own form-concatenation
///   intersection, applied position by position rather than once to a
///   whole set).
///
/// Only the TRUE arm narrows: `all(...)` being FALSE states only that
/// SOME element fails the predicate, not which one — no single window
/// characterizes that (the same "no complement" default `narrow_regex_
/// module_call`/`narrow_ascii_case_conjunction` already take). Any other
/// call shape (not `all`, not exactly one generator-expression argument,
/// more than one `for` clause, an `async for`, a filtered generator, an
/// iterable that is not a bare tracked name) narrows nothing — the
/// honest default every leaf in this file keeps.
pub(super) fn narrow_all_generator_call(call: &ruff_python_ast::ExprCall, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>, truth: bool) {
    if !truth {
        return;
    }
    let Some((predicate, loop_var, iterable_name)) = all_generator_shape(call) else {
        return;
    };
    let Some(current) = environment.read(iterable_name).cloned() else {
        return;
    };
    match current.kind {
        Kind::Set if current.set_kind_tag == SetKindTag::None => {
            let Some(repeated) = refined_sets::repetition_window_forms::as_repetition(&current.set) else {
                return;
            };
            let element = AbstractValue {
                kind_tag: current.kind_tag,
                ..known_set(repeated.element.clone(), None, TrustSpec, SetKindTag::None)
            };
            let narrowed_element = narrowed_via_predicate(predicate, loop_var, element, kernel);
            if narrowed_element.kind != Kind::Set {
                return;
            }
            let grade = trust_level_of(&current);
            let narrowed_set = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
                narrowed_element.set,
                repeated.lo,
                repeated.hi,
            )]);
            environment.bind(
                iterable_name,
                AbstractValue {
                    kind_tag: current.kind_tag,
                    ..known_set(narrowed_set, None, grade, current.set_kind_tag)
                },
            );
        }
        Kind::List => {
            let mut items = current.items.clone();
            let mut changed = false;
            for item in items.iter_mut() {
                if item.kind != Kind::Set {
                    continue;
                }
                let seed = AbstractValue {
                    kind_tag: item.kind_tag,
                    ..known_set(item.set.clone(), None, TrustSpec, SetKindTag::None)
                };
                let narrowed = narrowed_via_predicate(predicate, loop_var, seed, kernel);
                if narrowed.kind != Kind::Set || narrowed == *item {
                    continue;
                }
                let narrowed_item = meet_set_answer(item, &narrowed.set);
                *item = narrowed_item;
                changed = true;
            }
            if changed {
                environment.bind(iterable_name, refined_domain::known_constructors::known_list(items, trust_level_of(&current)));
            }
        }
        _ => {}
    }
}

/// Whether `call` is `all(<predicate> for <var> in <name>)`: the builtin
/// name `all`, exactly one positional argument that is itself a
/// generator expression (`Expr::Generator` — the same node a
/// parenthesis-free `all(... for ... in ...)` parses to, matching
/// `expressions.rs`'s own `Expr::Generator` arm), with exactly one
/// synchronous clause, no `if` filters (a filtered generator states a
/// narrower claim than plain iteration proves — declined rather than
/// overclaimed, mirroring `comprehension_target_and_star_element`'s own
/// single-clause restriction), a bare-name loop target, and a bare-name
/// iterable. The predicate expression, the loop variable's own name, and
/// the iterable's own name — or `None` for any other call shape.
pub(super) fn all_generator_shape(call: &ruff_python_ast::ExprCall) -> Option<(&Expr, &str, &str)> {
    let Expr::Name(func_name) = call.func.as_ref() else {
        return None;
    };
    if func_name.id.as_str() != "all" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [Expr::Generator(generator)] = &*call.arguments.args else {
        return None;
    };
    let [clause] = &*generator.generators else {
        return None;
    };
    if clause.is_async || !clause.ifs.is_empty() {
        return None;
    }
    let loop_var = name_of(&clause.target)?;
    let iterable_name = name_of(&clause.iter)?;
    Some((generator.elt.as_ref(), loop_var, iterable_name))
}

/// Asks what `predicate` being TRUE proves about `var`, starting from
/// `seed` (a fresh `Kind::Set` abstraction for one drawn element) — the
/// element-window twin of `guard_narrowed_values`'s own sandbox-narrow
/// pattern, generalized to read back a `Kind::Set` result rather than a
/// `Kind::Values` one (an element's own window is a RANGE, never an
/// enumerated list). Runs both of `assume`'s own name-keyed channels
/// (`narrow` then `narrow_set_kind_names`) exactly as `assume` itself
/// does, since the SET channel is what a numeric comparison over a
/// freshly-seeded `Kind::Set` binding narrows through. Returns `seed`
/// UNCHANGED when `var`'s own binding after the ask is not `Kind::Set`
/// (the predicate rebound it to something this reader does not
/// recognize) or is not itself bound — the honest "narrows nothing"
/// default, read by the caller as "no claim" via the equality check at
/// each call site.
pub(super) fn narrowed_via_predicate(predicate: &Expr, var: &str, seed: AbstractValue, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let mut sandbox = Environment::new(std::collections::HashSet::new());
    sandbox.bind(var, seed.clone());
    narrow(predicate, &mut sandbox, kernel, true);
    narrow_set_kind_names(predicate, &mut sandbox, kernel, true);
    match sandbox.read(var) {
        Some(bound) if bound.kind == Kind::Set => bound.clone(),
        _ => seed,
    }
}
