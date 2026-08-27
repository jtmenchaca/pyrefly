//! Expression evaluation into abstract values: literals, name reads
//! from the environment, unary minus, and arithmetic whose CPython row
//! is cited in PYREFLY-NUMERIC-B3-B4.md. This file is the contract the
//! walk calls; the expressions unit fills it in construct by construct.
//!
//! The re-export block below is this module's one door: every row its
//! children implement is named there, whether or not a caller outside
//! the module reads that particular row today. A row with no current
//! reader is still part of the stated interface, so the block carries
//! `allow(unused_imports)` rather than being trimmed to today's
//! callers and re-grown one line at a time as callers appear.
#![allow(unused_imports)]

mod literals;
mod subscript;
mod compare;
mod boolop_ternary;
mod fstring;
mod call;
mod attribute;
mod datetime;
mod json_re;
mod comprehension;
mod arithmetic;
mod sequence_ops;

#[cfg(test)]
mod tests;

// Test module is a sibling of the domain children, so re-export their
// items into this module's namespace for `tests`'s `use super::*`.
#[cfg(test)]
pub(super) use arithmetic::*;
#[cfg(test)]
pub(super) use attribute::*;
#[cfg(test)]
pub(super) use boolop_ternary::*;
#[cfg(test)]
pub(super) use call::*;
#[cfg(test)]
pub(super) use compare::*;
#[cfg(test)]
pub(super) use comprehension::*;
#[cfg(test)]
pub(super) use datetime::*;
#[cfg(test)]
pub(super) use fstring::*;
#[cfg(test)]
pub(super) use json_re::*;
#[cfg(test)]
pub(super) use literals::*;
#[cfg(test)]
pub(super) use sequence_ops::*;
#[cfg(test)]
pub(super) use subscript::*;

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;

use crate::env;
use crate::env::Environment;
use crate::string_models;

use arithmetic::evaluate_binop;
use literals::evaluate_unary;
use attribute::evaluate_attribute_read;
use compare::evaluate_boolop;
use fstring::evaluate_ternary;
use call::evaluate_call;
use subscript::evaluate_compare;
use comprehension::evaluate_dict_comp;
use comprehension::evaluate_list_or_set_comp;
use fstring::evaluate_fstring;
use call::evaluate_bytes_literal;
use literals::evaluate_dict;
use literals::evaluate_list;
use literals::evaluate_set;
use literals::evaluate_tuple;
use comprehension::number_literal_value;
use literals::evaluate_subscript;

pub use arithmetic::binary_arithmetic_value;
pub use datetime::binary_arithmetic_value_with_kernel;
pub use datetime::exact_instant_microseconds_of_expression;
pub use datetime::instant_stepped_by_microseconds;
pub use datetime::timedelta_microseconds_of_expression;
pub use datetime::utc_iso_microseconds;
pub use arithmetic::possible_raise;
pub(crate) use arithmetic::raised_exception_class;
pub use sequence_ops::provable_raise;
pub(crate) use call::call_one_argument_expression;
pub(crate) use call::exception_construction_value;
pub(crate) use call::fieldless_exception_value;
pub(crate) use call::math_from_imports;
pub use call::register_retained_callables;
pub use call::unmodeled_module_call_name;
pub(crate) use datetime::datetime_imports;
pub(crate) use datetime::module_never_calls_setlocale;
pub use datetime::DatetimeImports;
pub(crate) use subscript::slice_bound_index;

/// What this expression evaluates to in this environment. `unknown()`
/// is the honest default for every construct not yet built — an
/// unknown never fires and never silently passes a judgment.
pub fn evaluate_expression(
    expression: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    // A node whose value the walk already proved answers it directly
    // (`Environment::evaluated_node`). The relational sum is the one
    // publisher: a division whose operands the kernel tied together
    // answers more tightly than evaluating the two sides here could,
    // because the tie is a fact of the kernel program rather than of
    // either side. Checked at this one dispatch head, so a published
    // node is found wherever it sits in the tree.
    if let Some(published) = environment.evaluated_node(expression.range()) {
        return published.clone();
    }
    // THE DISPATCH SEAM (DERIVATION-TRACE.md, "Threading: dispatchers,
    // not readers"): one span per evaluated sub-expression, opened here
    // and closed when this scope drops, so every reader beneath keeps its
    // own signature and never manages a span. `refinery.construct` is
    // this node's OWN source spelling and `refinery.range` its own range
    // — never the enclosing statement's, which is exactly what makes the
    // deepest declined span name the blocking sub-expression.
    //
    // Off, this is one thread-local `Cell<bool>` read per node and
    // nothing else.
    let _span = crate::trace::span_scope(
        expression_reader_id(expression),
        usize::from(expression.range().start()),
        usize::from(expression.range().end()),
    );
    // THE BINDING LEDGER's lookup key on this node, where the node names
    // a place at all (`env::tracked_place_of`: a bare name, or a chain of
    // attribute reads over one). A read that ends up the trace's deepest
    // decline is exactly the bare-name leaf the ledger reclaims a binding
    // derivation for, and the tag is what says WHICH place to reclaim.
    // THE BINDING LEDGER's lookup key on this node, where the node names
    // a place at all (`env::tracked_place_of`: a bare name, or a chain of
    // attribute reads over one). A read that ends up the trace's deepest
    // decline is exactly the bare-name leaf the ledger reclaims a binding
    // derivation for, and the tag is what says WHICH place to reclaim.
    if crate::trace::is_tracing() {
        if let Some(place) = crate::env::tracked_place_of(expression) {
            crate::trace::record_read_place(&place.words());
        }
    }
    let value = evaluate_expression_dispatch(expression, environment, kernel);
    // What this node derived, in the kernel's own diagnostic spelling —
    // recorded for an ANSWERED node only. A node whose value is unknown
    // declined: it is a candidate blocker, and the decline helper at the
    // judging seam names the gate.
    if crate::trace::is_tracing() {
        record_expression_outcome(expression, environment, &value);
    }
    // Recorded ONLY when a caller asked for it (env.rs's own doc on
    // `evaluations`/`record_evaluation`) — an ordinary check never
    // opts in, so this is a no-op `Option` check for every node on
    // every walk except `check.rs::refined_set_at_position`'s own.
    environment.record_evaluation(expression.range(), value.clone());
    value
}

/// The adapter-local reader id for one expression form — the `name` its
/// dispatch span carries. Named after the arm of
/// `evaluate_expression_dispatch` that owns the form, so a trace reader
/// can go straight from a span to the code that produced it.
fn expression_reader_id(expression: &Expr) -> &'static str {
    match expression {
        Expr::NumberLiteral(_) => "number_literal",
        Expr::BooleanLiteral(_) => "boolean_literal",
        Expr::NoneLiteral(_) => "none_literal",
        Expr::StringLiteral(_) => "string_literal",
        Expr::BytesLiteral(_) => "bytes_literal",
        Expr::Name(_) => "name_read",
        Expr::UnaryOp(_) => "evaluate_unary",
        Expr::BinOp(_) => "evaluate_binop",
        Expr::List(_) => "evaluate_list",
        Expr::Set(_) => "evaluate_set",
        Expr::Tuple(_) => "evaluate_tuple",
        Expr::Dict(_) => "evaluate_dict",
        Expr::Subscript(_) => "evaluate_subscript",
        Expr::Compare(_) => "evaluate_compare",
        Expr::BoolOp(_) => "evaluate_boolop",
        Expr::FString(_) => "evaluate_fstring",
        Expr::If(_) => "evaluate_ternary",
        Expr::Named(_) => "walrus_read",
        Expr::Call(_) => "evaluate_call",
        Expr::Attribute(_) => "evaluate_attribute_read",
        Expr::ListComp(_) => "evaluate_list_comp",
        Expr::SetComp(_) => "evaluate_set_comp",
        Expr::Generator(_) => "evaluate_generator_comp",
        Expr::DictComp(_) => "evaluate_dict_comp",
        Expr::Await(_) => "await_read",
        Expr::Lambda(_) => "lambda_read",
        _ => "unread_expression_form",
    }
}

/// Records what one evaluated node derived onto its own open span: an
/// ANSWER for a value this domain actually read, and a DECLINE naming the
/// gate for a value it did not.
///
/// Sets and windows are spelled by `refined_sets::format_for_diagnostics`
/// — the kernel's own diagnostic formatter, the one spelling all three
/// adapters share (DERIVATION-TRACE.md's attribute-vocabulary rule). A
/// value shape with no kernel spelling states its own word, and that is a
/// gap to close rather than a convention.
fn record_expression_outcome(expression: &Expr, environment: &Environment, value: &AbstractValue) {
    use refined_domain::abstract_value::Kind;
    let spelled = spelled_value(value);
    match value.kind {
        // A NAME THIS BODY NEVER BINDS reading unbound is not a lost
        // binding: `time` in `time.monotonic_ns()` names an imported
        // module, and a module is never a refined value in the first
        // place. Recording it as a decline would make the trace's leaf
        // blame the import statement instead of the call whose model is
        // actually missing — the wrong construct to hand a reader as the
        // work item. `alias_is_visible` is the same "the body never
        // rebinds this name" test the module-level alias table already
        // uses, applied here to tell an outer name apart from a local
        // whose own binding statement derived nothing.
        Kind::Unknown if is_never_bound_outer_name(expression, environment) => {
            crate::trace::record_answer("a name this body never binds — an imported module or an outer global")
        }
        // The one shape that carries no reading at all — every reader
        // that could have answered this node declined, so the node itself
        // declines and becomes a blocker candidate.
        Kind::Unknown => crate::trace::record_decline(&unknown_gate(expression), None, Some(&spelled)),
        _ => crate::trace::record_answer(&spelled),
    }
}

/// Whether this expression is a bare Name that reads unbound AND that
/// this body never binds anywhere — an imported module name, or an outer
/// global. The same `alias_is_visible` test the module-level alias table
/// uses for "the body never rebinds this name".
fn is_never_bound_outer_name(expression: &Expr, environment: &Environment) -> bool {
    let Expr::Name(name) = expression else {
        return false;
    };
    environment.read(name.id.as_str()).is_none() && environment.alias_is_visible(name.id.as_str())
}

/// The named gate an UNREAD node earns, by its own syntactic form — the
/// distinction a reader of the trace needs, since "nothing derived this"
/// means something different for each form. A bare NAME that derived
/// nothing means its own binding never carried a value to this point,
/// which is the construct to go fix; a CALL that derived nothing means no
/// model answered that callee; and so on.
fn unknown_gate(expression: &Expr) -> String {
    match expression {
        Expr::Name(name) => format!(
            "'{}' carries no derived value at this read — the statement that binds it was never read into a set",
            name.id.as_str()
        ),
        Expr::Call(_) => {
            "no model answers this call, so it derives no value".to_owned()
        }
        Expr::Attribute(attribute) => format!(
            "reading '.{}' off this receiver derives no value",
            attribute.attr.as_str()
        ),
        Expr::BinOp(_) => "no arithmetic transfer answers this operator over these operands".to_owned(),
        Expr::Subscript(_) => "this subscript read derives no value".to_owned(),
        _ => "this expression's value was never derived by any reader".to_owned(),
    }
}

/// One abstract value in the kernel's own diagnostic spelling, where a
/// kernel formatter covers its shape. A `Kind::Set`'s refined set and a
/// `Kind::Values`' exact word both go through
/// `format_for_diagnostics`; the shapes that carry no refined set of
/// their own (an object, a list, `None`) state their own word.
pub(crate) fn spelled_value(value: &AbstractValue) -> String {
    use refined_domain::abstract_value::Kind;
    match value.kind {
        Kind::Set => refined_sets::format_for_diagnostics::format_for_diagnostics(&value.set),
        // An exact word: a string value IS its codepoint tuple, spelled
        // back as text; a numeric value is spelled the Python way. The
        // same two spellings `assignability::judge` puts in its own fire
        // messages, so a trace and a fire never name one value two ways.
        Kind::Values if value.kind_tag == Some(PrimitiveKind::String) => {
            refined_sets::format_string_shapes::from_points(&value.values)
                .unwrap_or_else(|| "a string value".to_owned())
        }
        Kind::Values => {
            let is_float = value.kind_tag == Some(PrimitiveKind::Float);
            let words: Vec<String> = value
                .values
                .iter()
                .map(|v| refined_sets::format_string_shapes::format_py_number(*v, is_float))
                .collect();
            match words.len() {
                1 => words.into_iter().next().expect("length checked"),
                _ => format!("{{{}}}", words.join(", ")),
            }
        }
        // An object states its own kind word where it has one. A TEMPORAL
        // construction carries its window instead — the calendar window
        // IS its refined reading, so spelling it "a dict" (the plain
        // dict-literal word) would name the carrier and hide the content.
        Kind::Object => match (value.kind_word, &value.temporal) {
            (Some(word), _) => word.to_owned(),
            // `format_temporal` is the ONE existing spelling for a
            // calendar window (`assignability::temporal`'s own refutation
            // sentences spell both sides with it) — never a second one
            // invented here.
            (None, Some(temporal)) => refined_sets::calendar_interpreter::format_temporal(temporal),
            (None, None) => "a dict".to_owned(),
        },
        Kind::List => "a list".to_owned(),
        Kind::Null => "None".to_owned(),
        Kind::NaN => "NaN".to_owned(),
        Kind::PossiblyNaN => "a value that may be NaN".to_owned(),
        Kind::PossiblyUndefined => "a value that may be absent".to_owned(),
        Kind::KindUnion => "a union of sorts".to_owned(),
        Kind::Unknown => "no reading".to_owned(),
        _ => "a value this domain carries no spelling for".to_owned(),
    }
}

fn evaluate_expression_dispatch(
    expression: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    match expression {
        // parenthesization carries no AST node of its own — ruff folds
        // `(x)` into `x` at parse time, so there is no case to write here
        Expr::NumberLiteral(literal) => number_literal_value(&literal.value),
        Expr::BooleanLiteral(literal) => {
            known_values(vec![if literal.value { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved)
        }
        // None is Python's one absent value — Kind::Null is the closest
        // faithful representation refined_domain carries (undef and null
        // both exist; None matches null_value's "the exactly-absent
        // marker" shape more than a wrapped maybe)
        Expr::NoneLiteral(_) => null_value(),
        // `__name__` is host-defined (the running module's own identity —
        // "__main__" when run as a script, the dotted module path
        // otherwise) but its SORT is always `str`
        // (tmp/cpython/Doc/reference/import.html#__name__ /
        // Doc/reference/datamodel.rst's module-attribute table both state
        // it as a string attribute) — a sort-only claim, never a specific
        // value, since this file has no host-execution-context knowledge
        // of which module is running. Only when the name is not locally
        // bound: an ordinary variable named `__name__` (shadowing the
        // module attribute) reads through the ordinary Name arm instead.
        Expr::Name(name) if name.id.as_str() == "__name__" && environment.read("__name__").is_none() => {
            known_set(strings(), None, TrustSpec, SetKindTag::None)
        }
        // A bare reference to a SAME-MODULE `def` — `f = identity`,
        // naming the function without calling it. `environment.read`
        // answers `None` here (a module-level `def` is indexed in
        // `environment.functions()`, never separately bound as a value
        // of its own), so without this arm the read would fall to the
        // catch-all `unknown()` below, discarding which function `f`
        // actually names and losing the call-through this value's later
        // `f(x)` needs (`env::same_module_def_alias_value`'s own doc).
        // Checked only when the name is not itself locally bound (a
        // local shadowing a def name reads its own binding instead, the
        // same shadow-on-rebind rule every other module-level fact in
        // this file keeps) and the module's function table actually
        // names a def there.
        Expr::Name(name) if environment.read(name.id.as_str()).is_none() && environment.functions().is_some_and(|table| table.def(name.id.as_str()).is_some()) => {
            env::same_module_def_alias_value(name.id.as_str())
        }
        Expr::Name(name) => match environment.read(name.id.as_str()) {
            Some(value) => value.clone(),
            None => unknown(),
        },
        Expr::UnaryOp(unary) => evaluate_unary(unary, environment, kernel),
        Expr::BinOp(binop) => evaluate_binop(binop, environment, kernel),
        Expr::StringLiteral(literal) => string_models::string_literal_value(literal.value.to_str()),
        Expr::BytesLiteral(literal) => evaluate_bytes_literal(literal),
        Expr::List(list) => evaluate_list(list, environment, kernel),
        Expr::Set(set) => evaluate_set(set, environment, kernel),
        Expr::Tuple(tuple) => evaluate_tuple(tuple, environment, kernel),
        Expr::Dict(dict) => evaluate_dict(dict, environment, kernel),
        Expr::Subscript(subscript) => evaluate_subscript(subscript, environment, kernel),
        Expr::Compare(compare) => evaluate_compare(compare, environment, kernel),
        Expr::BoolOp(boolop) => evaluate_boolop(boolop, environment, kernel),
        Expr::FString(fstring) => evaluate_fstring(fstring, environment, kernel),
        Expr::If(ternary) => evaluate_ternary(ternary, environment, kernel),
        Expr::Named(named) => evaluate_expression(&named.value, environment, kernel),
        Expr::Call(call) => evaluate_call(call, environment, kernel),
        Expr::Attribute(attribute) => evaluate_attribute_read(attribute, environment, kernel),
        Expr::ListComp(comp) => evaluate_list_or_set_comp(&comp.elt, &comp.generators, environment, kernel),
        Expr::SetComp(comp) => evaluate_list_or_set_comp(&comp.elt, &comp.generators, environment, kernel),
        Expr::Generator(comp) => evaluate_list_or_set_comp(&comp.elt, &comp.generators, environment, kernel),
        Expr::DictComp(comp) => evaluate_dict_comp(comp, environment, kernel),
        Expr::Await(inner) => evaluate_expression(&inner.value, environment, kernel),
        // `lambda: ...` read as a VALUE (bound to a name, returned, or
        // otherwise used directly rather than called) — expressions.rst,
        // "Lambdas": "The expression `lambda parameters: expression`
        // yields a function object." The unnamed object behaves like an
        // ordinary `def`-built function object (datamodel.rst,
        // "User-defined functions"). This domain tracks no
        // function-value Kind (a
        // function is never itself a refined scalar/collection), so the
        // honest answer is opaque — "a function value," never a
        // specific scalar (b-body-expressions.py's
        // `function_stored_as_local`).
        //
        // RETAINED CALLABLE: when `register_retained_callables` has
        // already recorded a creation of this exact lambda into
        // `environment` (its statement-level caller runs that scan
        // before reaching this evaluation — `check.rs::sink_value`,
        // `summaries::interpret_body`'s `Stmt::Return` arm), the value
        // encodes the CURRENT retained-callable key on `source`
        // (`env::retained_callable_value`) so a later call through
        // `evaluate_call`'s retained-callable arm can interpret the
        // body instead of declining. The key is read back through
        // `environment.lambda_key` — never the range itself as a key
        // (`env.rs`'s own doc on why a fresh id is minted per creation,
        // not the AST range) — so two creations of the SAME lambda
        // text (`make_adder(1)` and `make_adder(100)`, each closing
        // over a different `step`) never conflate. A lambda `register_
        // retained_callables` never reached (a shape outside its own
        // recursion, or an environment with no such registration step
        // at all — every existing test environment, unaffected) still
        // answers the plain opaque value exactly as before this table
        // existed.
        Expr::Lambda(lambda) => match environment.lambda_key(lambda.range().start().to_u32()) {
            Some(key) => env::retained_callable_value(key),
            None => opaque_value("a function value"),
        },
        _ => unknown(),
    }
}
