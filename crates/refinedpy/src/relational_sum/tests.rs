use super::*;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::loop_questions::stmt_wire;
use refined_kernel::loop_questions::IrStatement;
use refined_kernel::loop_questions::IrStatementKind;
use refined_kernel::loop_questions::LoopEffect;
use refined_kernel::loop_questions::LoopEffectKind;
use refined_kernel::loop_questions::LoopEffectOp;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;

use crate::env::Environment;
use ruff_text_size::Ranged;

// `total += s * s` over an element slot — the fixture's own body
// (audio_level.py:19), lowered.
fn squared_element() -> LoopEffect {
    LoopEffect {
        kind: LoopEffectKind::Binary,
        op: LoopEffectOp::Mul,
        a: Some(Box::new(slot(ELEMENT_SLOT))),
        b: Some(Box::new(slot(ELEMENT_SLOT))),
        ..Default::default()
    }
}

#[test]
fn the_accumulation_lowers_to_the_loop_accum_statement() {
    let statement = IrStatement {
        kind: IrStatementKind::LoopAccum,
        target: TOTAL_SLOT,
        accum_src: ELEMENT_SLOT,
        accum_len: COUNT_SLOT,
        effect: squared_element(),
        ..Default::default()
    };
    let got = stmt_wire(&statement);
    let want = r#"{"loopAccum":{"total":0,"src":1,"len":2,"body":{"op":"binary64.mul","A":{"var":1},"B":{"var":1}}}}"#;
    assert_eq!(got, want, "stmt_wire(accumulation) = {got:?}, want {want:?}");
}

#[test]
fn the_folded_division_divides_the_total_slot_by_the_count_slot() {
    let mut recognized = RecognizedAccumulation {
        total_name: "total".to_owned(),
        sequence_name: "samples".to_owned(),
        entry_states: Vec::new(),
        statements: Vec::new(),
        grade: TrustProved,
        total_kind_tag: None,
        quotient_op: None,
        length_aliases: std::collections::HashMap::new(),
        count_set: make_refined_set(vec![]),
    };
    let expression = division_expression("total", "samples");
    assert!(
        fold_division(&mut recognized, &expression, &environment_with_samples()),
        "fold_division declined `total / len(samples)`"
    );
    let [division] = recognized.statements.as_slice() else {
        panic!("want exactly the division statement, got {:?}", recognized.statements.len());
    };
    let got = stmt_wire(division);
    // the quotient lands in slot 3, its own — the total's slot 0 is
    // read as the numerator and left holding the total
    let want = r#"{"assign":{"target":3,"e":{"op":"binary64.div","A":{"var":0},"B":{"var":2}}}}"#;
    assert_eq!(got, want, "stmt_wire(division) = {got:?}, want {want:?}");
}

#[test]
fn a_division_by_another_sequences_length_is_not_folded() {
    let mut recognized = RecognizedAccumulation {
        total_name: "total".to_owned(),
        sequence_name: "samples".to_owned(),
        entry_states: Vec::new(),
        statements: Vec::new(),
        grade: TrustProved,
        total_kind_tag: None,
        quotient_op: None,
        length_aliases: std::collections::HashMap::new(),
        count_set: make_refined_set(vec![]),
    };
    let expression = division_expression("total", "others");
    assert!(
        !fold_division(&mut recognized, &expression, &environment_with_samples()),
        "fold_division accepted a length taken of a different sequence"
    );
    assert!(
        recognized.statements.is_empty(),
        "a declined division must leave the program alone"
    );
}

#[test]
fn is_length_alias_assignment_recognizes_count_equals_len_of_the_sequence() {
    let recognized = recognized_over_samples();
    let assign = parsed_assignment("count = len(samples)\n");
    assert_eq!(
        is_length_alias_assignment(&assign, &recognized, &environment_with_samples()),
        Some("count".to_owned()),
        "a plain `count = len(samples)` must be read as the count-alias shape"
    );
}

#[test]
fn is_length_alias_assignment_declines_a_different_sequences_length() {
    let recognized = recognized_over_samples();
    let assign = parsed_assignment("count = len(others)\n");
    assert_eq!(
        is_length_alias_assignment(&assign, &recognized, &environment_with_samples()),
        None,
        "a length taken of a different sequence is not this accumulation's count"
    );
}

// PIN (ledger 218): the aliased spelling — `count = len(samples)`
// then `total / count` — folds to the IDENTICAL division statement
// `total / len(samples)` folds to. This test drives `relational_sum`'s
// own API (`is_length_alias_assignment`, `record_length_alias`,
// `fold_division`) directly; wiring `check.rs`'s own statement scan to
// call that API — the one-hop loop over `following` that currently
// inspects only `following.first()` (check.rs:2778-2787) — is the
// dependency this pin does not itself exercise. See the report for
// the exact check.rs change.
#[test]
fn the_count_alias_shape_folds_identically_to_the_direct_spelling() {
    let mut aliased = recognized_over_samples();
    let alias_assignment = parsed_assignment("count = len(samples)\n");
    let alias = is_length_alias_assignment(&alias_assignment, &aliased, &environment_with_samples())
        .expect("count = len(samples) is the count-alias shape");
    record_length_alias(&mut aliased, alias);
    let aliased_expression = bare_division_expression("total", "count");
    assert!(
        fold_division(&mut aliased, &aliased_expression, &environment_with_samples()),
        "fold_division declined the recorded count alias"
    );

    let mut direct = recognized_over_samples();
    let direct_expression = division_expression("total", "samples");
    assert!(
        fold_division(&mut direct, &direct_expression, &environment_with_samples()),
        "fold_division declined the direct len() spelling"
    );

    let [aliased_statement] = aliased.statements.as_slice() else {
        panic!("want exactly the division statement, got {:?}", aliased.statements.len());
    };
    let [direct_statement] = direct.statements.as_slice() else {
        panic!("want exactly the division statement, got {:?}", direct.statements.len());
    };
    assert_eq!(
        stmt_wire(aliased_statement),
        stmt_wire(direct_statement),
        "the aliased spelling must lower to the same statement the direct spelling does"
    );
}

#[test]
fn an_unrecorded_alias_still_declines() {
    // `record_length_alias` never ran — the same name with no
    // recorded link must decline exactly as an unrelated name does
    let mut recognized = recognized_over_samples();
    let expression = bare_division_expression("total", "count");
    assert!(
        !fold_division(&mut recognized, &expression, &environment_with_samples()),
        "an alias with no recorded link must not fold"
    );
}

#[test]
fn reassigns_alias_or_sequence_catches_an_assign_augassign_and_for_target() {
    let assign = parsed_assignment("count = 0\n");
    assert!(
        reassigns_alias_or_sequence(&Stmt::Assign(assign), "count", "samples"),
        "an Assign target naming the alias must be caught"
    );
    let module = ruff_python_parser::parse_module("samples += extra\n")
        .expect("the test's own source parses")
        .into_syntax();
    let aug = module.body.into_iter().next().expect("one statement");
    assert!(
        reassigns_alias_or_sequence(&aug, "count", "samples"),
        "an AugAssign target naming the sequence must be caught"
    );
    let module = ruff_python_parser::parse_module("for samples in other:\n    pass\n")
        .expect("the test's own source parses")
        .into_syntax();
    let for_stmt = module.body.into_iter().next().expect("one statement");
    assert!(
        reassigns_alias_or_sequence(&for_stmt, "count", "samples"),
        "a for-loop target naming the sequence must be caught"
    );
}

#[test]
fn reassigns_alias_or_sequence_ignores_an_unrelated_statement() {
    let module = ruff_python_parser::parse_module("other = 0\n")
        .expect("the test's own source parses")
        .into_syntax();
    let unrelated = module.body.into_iter().next().expect("one statement");
    assert!(
        !reassigns_alias_or_sequence(&unrelated, "count", "samples"),
        "a statement touching neither watched name must not be caught"
    );
}

#[test]
fn a_division_by_a_1_to_1_comprehensions_length_is_folded() {
    // `clamped = [max(-1.0, min(1.0, s)) for s in samples]` proves
    // `len(clamped) == len(samples)` exactly, so `total / len(clamped)`
    // ties to the SAME accumulation the loop over `samples` ran.
    let mut recognized = RecognizedAccumulation {
        total_name: "total".to_owned(),
        sequence_name: "samples".to_owned(),
        entry_states: Vec::new(),
        statements: Vec::new(),
        grade: TrustProved,
        total_kind_tag: None,
        quotient_op: None,
        length_aliases: std::collections::HashMap::new(),
        count_set: make_refined_set(vec![]),
    };
    let mut environment = environment_with_samples();
    environment.bind(
        "clamped",
        AbstractValue {
            same_length_as: Some("samples".to_owned()),
            ..known_set(
                make_refined_set(vec![refined_sets::refinement_forms::star(
                    make_refined_set(vec![at_least(-1.0), at_most(1.0)]),
                )]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        },
    );
    let expression = division_expression("total", "clamped");
    assert!(
        fold_division(&mut recognized, &expression, &environment),
        "fold_division declined a length proved equal via same_length_as"
    );
}

#[test]
fn a_division_by_a_filtered_comprehensions_length_is_not_folded() {
    // a filtered comprehension's own builder (expressions.rs's
    // `comprehension_star_elements`) leaves `same_length_as` unset —
    // this pins the CONSUMING side of that same soundness line: an
    // unlinked name still declines, exactly as an unrelated name does.
    let mut recognized = RecognizedAccumulation {
        total_name: "total".to_owned(),
        sequence_name: "samples".to_owned(),
        entry_states: Vec::new(),
        statements: Vec::new(),
        grade: TrustProved,
        total_kind_tag: None,
        quotient_op: None,
        length_aliases: std::collections::HashMap::new(),
        count_set: make_refined_set(vec![]),
    };
    let mut environment = environment_with_samples();
    environment.bind(
        "positives",
        known_set(
            make_refined_set(vec![refined_sets::refinement_forms::star(
                make_refined_set(vec![at_least(0.0), at_most(1.0)]),
            )]),
            None,
            TrustProved,
            SetKindTag::None,
        ),
    );
    let expression = division_expression("total", "positives");
    assert!(
        !fold_division(&mut recognized, &expression, &environment),
        "fold_division accepted a length with no proved link"
    );
    assert!(
        recognized.statements.is_empty(),
        "a declined division must leave the program alone"
    );
}

// `<numerator> / len(<sequence>)`, parsed rather than hand-built:
// the AST shapes carry ranges and parenthesization this module
// reads through, so a parsed expression is the honest input.
fn division_expression(numerator: &str, sequence: &str) -> Expr {
    let source = format!("{numerator} / len({sequence})");
    let parsed = ruff_python_parser::parse_expression(&source)
        .expect("the test's own source parses");
    *parsed.into_syntax().body
}

// `<numerator> / <divisor>`, with the divisor a BARE name — the
// count-alias shape (`total / count`), never wrapped in `len(...)`
// the way `division_expression` always wraps its second argument.
fn bare_division_expression(numerator: &str, divisor: &str) -> Expr {
    let source = format!("{numerator} / {divisor}");
    let parsed = ruff_python_parser::parse_expression(&source)
        .expect("the test's own source parses");
    *parsed.into_syntax().body
}

// A module whose single statement is the assignment under test.
fn parsed_assignment(source: &str) -> StmtAssign {
    let module = ruff_python_parser::parse_module(source)
        .expect("the test's own source parses")
        .into_syntax();
    let Some(Stmt::Assign(assign)) = module.body.into_iter().next() else {
        panic!("the test's own source must be a single assignment");
    };
    assign
}

// An environment holding `samples` as an unknown-length sequence of
// -1.0 … 1.0 — the star shape `seed_parameters` builds for a
// `Sequence[float]` parameter, and what the fixture's own body sees.
fn environment_with_samples() -> Environment {
    let element = make_refined_set(vec![at_least(-1.0), at_most(1.0)]);
    let mut environment = Environment::new(std::collections::HashSet::from([
        "total".to_owned(),
        "samples".to_owned(),
    ]));
    environment.bind(
        "samples",
        known_set(
            make_refined_set(vec![refined_sets::refinement_forms::star(element)]),
            None,
            TrustProved,
            SetKindTag::None,
        ),
    );
    environment
}

#[test]
fn the_generator_sum_lowers_to_the_same_program_the_explicit_loop_does() {
    // the fixture's own statement (audio_level.py:19) — `s * s` is
    // the SAME source variable on both sides, so this lowers to
    // the structural `sq` effect, not the general `mul` of two
    // vars.
    let assign = parsed_assignment("total = sum(s * s for s in samples)\n");
    let recognized = recognize_generator_sum(&assign, &environment_with_samples())
        .expect("the generator sum over a star sequence recognizes");
    assert_eq!(recognized.total_name, "total");
    assert_eq!(recognized.sequence_name, "samples");
    let [accumulation] = recognized.statements.as_slice() else {
        panic!("want exactly the accumulation, got {}", recognized.statements.len());
    };
    let got = stmt_wire(accumulation);
    let want = r#"{"loopAccum":{"total":0,"src":1,"len":2,"body":{"sq":1}}}"#;
    assert_eq!(got, want, "stmt_wire(generator sum) = {got:?}, want {want:?}");
}

#[test]
fn the_generator_sums_entry_states_start_the_total_at_zero() {
    let assign = parsed_assignment("total = sum(s * s for s in samples)\n");
    let recognized = recognize_generator_sum(&assign, &environment_with_samples())
        .expect("the generator sum recognizes");
    assert_eq!(recognized.entry_states.len(), 4, "want one state per slot");
    let total = &recognized.entry_states[TOTAL_SLOT as usize];
    assert_eq!(
        total.set,
        make_refined_set(vec![one_of(&[0.0])]),
        "the total enters at exactly 0, want {:?}",
        total.set
    );
    assert!(
        recognized.entry_states[QUOTIENT_SLOT as usize].top,
        "the quotient slot holds nothing until a division writes it"
    );
}

#[test]
fn a_generator_sum_with_a_nonzero_start_is_declined() {
    // the generator is parenthesized because a bare one cannot be
    // followed by another argument (expressions.rst, "Generator
    // expressions")
    let assign = parsed_assignment("total = sum((s * s for s in samples), 5)\n");
    assert!(
        recognize_generator_sum(&assign, &environment_with_samples()).is_none(),
        "a nonzero start shifts the total off the relation's zero base"
    );
}

#[test]
fn a_generator_sum_with_an_explicit_zero_start_is_recognized() {
    let assign = parsed_assignment("total = sum((s * s for s in samples), 0)\n");
    assert!(
        recognize_generator_sum(&assign, &environment_with_samples()).is_some(),
        "an explicit start of 0 is sum's own default and stays recognized"
    );
}

#[test]
fn a_filtered_generator_is_declined() {
    // an `if` clause drops elements, so the count the relation ties
    // the total to is no longer the sequence's own length
    let assign = parsed_assignment("total = sum(s * s for s in samples if s > 0)\n");
    assert!(
        recognize_generator_sum(&assign, &environment_with_samples()).is_none(),
        "a filtered generator sums over an unstated count"
    );
}

#[test]
fn a_list_comprehension_argument_is_left_to_the_eager_path() {
    let assign = parsed_assignment("total = sum([s * s for s in samples])\n");
    assert!(
        recognize_generator_sum(&assign, &environment_with_samples()).is_none(),
        "sum([...]) is already materialized eagerly and must not be raced"
    );
}

#[test]
fn a_locally_bound_sum_is_not_the_builtin() {
    let assign = parsed_assignment("total = sum(s * s for s in samples)\n");
    let mut environment = environment_with_samples();
    environment.bind("sum", known_set(make_refined_set(vec![]), None, TrustProved, SetKindTag::None));
    assert!(
        recognize_generator_sum(&assign, &environment).is_none(),
        "a shadowed `sum` name is not the builtin this reader models"
    );
}

#[test]
fn a_generator_body_reading_a_free_name_is_declined() {
    let assign = parsed_assignment("total = sum(s * gain for s in samples)\n");
    assert!(
        recognize_generator_sum(&assign, &environment_with_samples()).is_none(),
        "a term reading a name outside the element cannot be lowered exactly"
    );
}

// An environment holding `xs` as an unknown-length sequence of
// Integer-sorted elements in [0, 9] — the star shape `seed_parameters`
// builds for a `list[int]` parameter (`B2.use.sink`'s own `xs:
// list[int]`, elements narrowed to [0, 150] by the fixture's own
// guard; [0, 9] here is an arbitrary bounded window, since
// `recognize_sum_over_name` itself never reads the element bounds,
// only the element SORT).
fn environment_with_integer_xs() -> Environment {
    let element = make_refined_set(vec![at_least(0.0), at_most(9.0)]);
    let mut environment =
        Environment::new(std::collections::HashSet::from(["total".to_owned(), "xs".to_owned()]));
    environment.bind(
        "xs",
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![refined_sets::refinement_forms::star(element)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        },
    );
    environment
}

#[test]
fn sum_over_a_plain_name_recognizes_and_lowers_to_loop_accum() {
    let assign = parsed_assignment("total = sum(xs)\n");
    let recognized = recognize_sum_over_name(&assign, &environment_with_integer_xs())
        .expect("sum(xs) over an Integer-sorted star sequence recognizes");
    assert_eq!(recognized.total_name, "total");
    assert_eq!(recognized.sequence_name, "xs");
    assert_eq!(
        recognized.total_kind_tag,
        Some(PrimitiveKind::Integer),
        "sum(xs) performs no per-element transform, so the total's sort is xs's own element sort"
    );
    let [accumulation] = recognized.statements.as_slice() else {
        panic!("want exactly the accumulation, got {}", recognized.statements.len());
    };
    let got = stmt_wire(accumulation);
    // no per-element transform: the per-trip effect is the bare
    // element slot, unlike the generator forms' own transformed body
    let want = r#"{"loopAccum":{"total":0,"src":1,"len":2,"body":{"var":1}}}"#;
    assert_eq!(got, want, "stmt_wire(sum over a plain name) = {got:?}, want {want:?}");
}

#[test]
fn sum_over_a_plain_name_with_an_explicit_zero_start_is_recognized() {
    let assign = parsed_assignment("total = sum(xs, 0)\n");
    assert!(
        recognize_sum_over_name(&assign, &environment_with_integer_xs()).is_some(),
        "an explicit start of 0 is sum's own default and stays recognized"
    );
}

#[test]
fn sum_over_a_plain_name_with_a_nonzero_start_is_declined() {
    let assign = parsed_assignment("total = sum(xs, 5)\n");
    assert!(
        recognize_sum_over_name(&assign, &environment_with_integer_xs()).is_none(),
        "a nonzero start shifts the total off the relation's zero base"
    );
}

#[test]
fn sum_over_a_generator_is_left_to_recognize_generator_sum() {
    // the ARGUMENT must be a bare name; a generator argument is the
    // other recognizer's own shape and must not double-match here
    let assign = parsed_assignment("total = sum(s * s for s in xs)\n");
    assert!(
        recognize_sum_over_name(&assign, &environment_with_integer_xs()).is_none(),
        "a generator argument is recognize_generator_sum's shape, not this one's"
    );
}

#[test]
fn sum_over_a_list_display_is_left_to_the_eager_path() {
    let assign = parsed_assignment("total = sum([1, 2, 3])\n");
    assert!(
        recognize_sum_over_name(&assign, &environment_with_integer_xs()).is_none(),
        "a list display argument is not a bare name and is already materialized eagerly"
    );
}

#[test]
fn sum_over_a_name_with_no_known_element_sort_is_declined() {
    // xs is bound, but with no kind_tag at all — the shape
    // `environment_with_samples`'s own `samples` binding takes
    let assign = parsed_assignment("total = sum(samples)\n");
    assert!(
        recognize_sum_over_name(&assign, &environment_with_samples()).is_none(),
        "the total's sort must be known exactly; an unset kind_tag states neither Integer nor Float"
    );
}

#[test]
fn a_locally_bound_sum_is_not_the_builtin_over_a_plain_name() {
    let assign = parsed_assignment("total = sum(xs)\n");
    let mut environment = environment_with_integer_xs();
    environment.bind("sum", known_set(make_refined_set(vec![]), None, TrustProved, SetKindTag::None));
    assert!(
        recognize_sum_over_name(&assign, &environment).is_none(),
        "a shadowed `sum` name is not the builtin this reader models"
    );
}

// A recognized accumulation over `total` and `samples`, with no
// program built — enough for the division readers, which only ever
// consult the two names.
fn recognized_over_samples() -> RecognizedAccumulation {
    RecognizedAccumulation {
        total_name: "total".to_owned(),
        sequence_name: "samples".to_owned(),
        entry_states: Vec::new(),
        statements: Vec::new(),
        grade: TrustProved,
        total_kind_tag: None,
        quotient_op: None,
        length_aliases: std::collections::HashMap::new(),
        count_set: make_refined_set(vec![]),
    }
}

// The expression of a `return <source>` inside a def — the position
// a return actually occupies, so the parse carries no diagnostic of
// its own.
fn returned_expression(source: &str) -> Expr {
    let module = ruff_python_parser::parse_module(&format!("def f():\n    return {source}\n"))
        .expect("the test's own source parses")
        .into_syntax();
    let Some(Stmt::FunctionDef(def)) = module.body.into_iter().next() else {
        panic!("the test's own source must be a single def");
    };
    let Some(Stmt::Return(ret)) = def.body.into_iter().next() else {
        panic!("the def's body must be a single return");
    };
    *ret.value.expect("the return carries a value")
}

#[test]
fn the_division_is_found_nested_inside_a_call_argument() {
    // the fixture's own return (audio_level.py:25)
    let returned = returned_expression("math.sqrt(total / len(samples))");
    let (range, op) = division_range_in(&returned, &recognized_over_samples(), &environment_with_samples())
        .expect("the nested division is found");
    // the located node is the inner division, strictly inside the
    // call that wraps it
    assert!(
        range.start() > returned.range().start(),
        "want the inner division's range, not the whole call's: {range:?}"
    );
    assert!(
        range.end() < returned.range().end(),
        "want the inner division's range, not the whole call's: {range:?}"
    );
    assert_eq!(op, DivisionOp::Div, "the source spelling is `/`, not `//`");
}

#[test]
fn a_bare_division_in_return_position_is_found() {
    let returned = returned_expression("total / len(samples)");
    let (range, op) = division_range_in(&returned, &recognized_over_samples(), &environment_with_samples())
        .expect("a top-level division is found");
    assert_eq!(
        range,
        returned.range(),
        "the whole returned expression IS the division"
    );
    assert_eq!(op, DivisionOp::Div, "the source spelling is `/`, not `//`");
}

#[test]
fn a_return_holding_two_divisions_is_declined() {
    // one published answer cannot say which node it belongs to
    let returned = returned_expression("total / len(samples) + total / len(samples)");
    assert!(
        division_range_in(&returned, &recognized_over_samples(), &environment_with_samples()).is_none(),
        "two occurrences must decline rather than pick one"
    );
}

#[test]
fn a_return_holding_no_division_is_declined() {
    let returned = returned_expression("math.sqrt(total)");
    assert!(
        division_range_in(&returned, &recognized_over_samples(), &environment_with_samples()).is_none(),
        "there is nothing to fold"
    );
}

#[test]
fn a_division_by_a_different_sequences_length_is_not_found() {
    let returned = returned_expression("math.sqrt(total / len(others))");
    assert!(
        division_range_in(&returned, &recognized_over_samples(), &environment_with_samples()).is_none(),
        "a length taken of another sequence carries no relation"
    );
}

#[test]
fn a_division_inside_a_lambda_body_is_not_this_accumulations() {
    // the lambda's body is its own scope, so its `total` is a
    // different binding
    let returned = returned_expression("lambda: total / len(samples)");
    assert!(
        division_range_in(&returned, &recognized_over_samples(), &environment_with_samples()).is_none(),
        "a lambda body is a separate scope"
    );
}

#[test]
fn the_located_division_folds_the_same_statement_the_assignment_form_does() {
    let mut recognized = recognized_over_samples();
    fold_located_division(&mut recognized, DivisionOp::Div);
    let [division] = recognized.statements.as_slice() else {
        panic!("want exactly the division statement, got {}", recognized.statements.len());
    };
    let got = stmt_wire(division);
    let want = r#"{"assign":{"target":3,"e":{"op":"binary64.div","A":{"var":0},"B":{"var":2}}}}"#;
    assert_eq!(got, want, "stmt_wire(located division) = {got:?}, want {want:?}");
    assert_eq!(
        recognized.quotient_op,
        Some(DivisionOp::Div),
        "fold_located_division must record which operator folded"
    );
}

#[test]
fn the_located_floor_division_wraps_the_div_effect_in_floor() {
    let mut recognized = recognized_over_samples();
    fold_located_division(&mut recognized, DivisionOp::FloorDiv);
    let [division] = recognized.statements.as_slice() else {
        panic!("want exactly the division statement, got {}", recognized.statements.len());
    };
    let got = stmt_wire(division);
    let want =
        r#"{"assign":{"target":3,"e":{"op":"binary64.floor","A":{"op":"binary64.div","A":{"var":0},"B":{"var":2}}}}}"#;
    assert_eq!(got, want, "stmt_wire(located floor division) = {got:?}, want {want:?}");
}

// PIN: Python's `//` on two int operands yields an int
// (expressions.rst, binary arithmetic) — `total // len(xs)` with
// `total` an int-sorted sum (`recognize_sum_over_name` over
// `list[int]`) must publish an Integer-sorted quotient, matching the
// `int(...)` sink the fixture assigns it to (`B2.use.sink`,
// `m = total // len(xs)`).
#[test]
fn int_sum_floor_divided_by_len_is_integer_sorted() {
    let mut recognized = recognized_over_samples();
    recognized.total_kind_tag = Some(PrimitiveKind::Integer);
    recognized.quotient_op = Some(DivisionOp::FloorDiv);
    assert_eq!(
        quotient_kind_tag(&recognized),
        Some(PrimitiveKind::Integer),
        "int // int must publish an Integer quotient"
    );
}

// PIN: Python's `/` yields a float unconditionally, and `//` over
// ANY float operand also yields a float — same clause. Both must
// keep publishing Float, exactly as the hardcoded rule did before
// FloorDiv folded at all (`B2`'s own `audio_level.py`-style
// `total / len(samples)` mean/sqrt rows).
#[test]
fn float_sum_divided_by_len_stays_float_sorted() {
    let mut float_true_div = recognized_over_samples();
    float_true_div.total_kind_tag = Some(PrimitiveKind::Float);
    float_true_div.quotient_op = Some(DivisionOp::Div);
    assert_eq!(
        quotient_kind_tag(&float_true_div),
        Some(PrimitiveKind::Float),
        "true division must stay Float regardless of the total's own sort"
    );

    let mut float_floor_div = recognized_over_samples();
    float_floor_div.total_kind_tag = Some(PrimitiveKind::Float);
    float_floor_div.quotient_op = Some(DivisionOp::FloorDiv);
    assert_eq!(
        quotient_kind_tag(&float_floor_div),
        Some(PrimitiveKind::Float),
        "float // int must stay Float — the total is not Integer-sorted"
    );

    let mut unknown_total = recognized_over_samples();
    unknown_total.total_kind_tag = None;
    unknown_total.quotient_op = Some(DivisionOp::FloorDiv);
    assert_eq!(
        quotient_kind_tag(&unknown_total),
        Some(PrimitiveKind::Float),
        "an unknown total sort must not be assumed Integer"
    );
}

#[test]
fn the_count_state_is_a_nonnegative_integer_bounded_by_the_window() {
    let element = make_refined_set(vec![at_least(-1.0), at_most(1.0)]);
    let sequence = known_set(
        make_refined_set(vec![refined_sets::refinement_forms::star(element.clone())]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let (read_element, count) =
        element_and_count_sets(&sequence).expect("a star sequence reads its window");
    assert_eq!(read_element, element, "the element set reads back unchanged");
    // the star window is (0, unbounded): a whole count, at least 0,
    // with no upper bound to state
    let want = make_refined_set(vec![integer(), at_least(0.0)]);
    assert_eq!(count, want, "count = {count:?}, want {want:?}");
}

#[test]
fn a_sequence_with_no_element_set_is_declined() {
    let empty = known_set(
        make_refined_set(vec![refined_sets::refinement_forms::star(
            make_refined_set(vec![]),
        )]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    assert!(
        element_and_count_sets(&empty).is_none(),
        "an element set stating nothing must decline"
    );
}

// A state carrying the given forms — never top, never absent/NaN/thrown —
// the shape a kernel exit state answering a plain number takes.
fn plain_number_state(forms: Vec<refined_sets::refinement_forms::Refinement>) -> KnownStateWire {
    number_state(make_refined_set(forms))
}

#[test]
fn a_full_range_state_binds_no_answer() {
    // the lone form `numbers()` itself spells — no lower bound, no
    // upper form at all — is exactly ℝ̄, no claim over plain top
    let state = plain_number_state(vec![at_least(f64::NEG_INFINITY)]);
    assert!(
        bindable_state(&state, TrustProved, None).is_none(),
        "a state spanning every float must answer no bindable value"
    );
}

#[test]
fn a_full_range_state_with_an_explicit_positive_infinity_ceiling_binds_no_answer() {
    let state = plain_number_state(vec![at_least(f64::NEG_INFINITY), at_most(f64::INFINITY)]);
    assert!(
        bindable_state(&state, TrustProved, None).is_none(),
        "an explicit +inf ceiling states no more than the ceiling's own absence"
    );
}

#[test]
fn a_bounded_state_still_binds() {
    let state = plain_number_state(vec![at_least(0.0), at_most(1.0)]);
    let bound = bindable_state(&state, TrustProved, None);
    assert!(bound.is_some(), "a genuinely bounded state must still bind");
}

#[test]
fn a_state_excluding_only_negative_infinity_is_not_the_full_range() {
    // `above(-inf)` is `x > -inf`, which excludes the single point
    // -inf — narrower than ℝ̄, so this must still bind
    let state = plain_number_state(vec![refined_sets::refinement_forms::above(f64::NEG_INFINITY)]);
    assert!(
        bindable_state(&state, TrustProved, None).is_some(),
        "excluding the point -inf is a real claim, not the full range"
    );
}

#[test]
fn a_full_ray_alongside_another_form_is_not_the_full_range() {
    // an Integer form riding alongside the full range still excludes
    // every non-integer float
    let state = plain_number_state(vec![at_least(f64::NEG_INFINITY), integer()]);
    assert!(
        bindable_state(&state, TrustProved, None).is_some(),
        "an Integer form alongside a full ray still states a real claim"
    );
}
