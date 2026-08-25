use super::*;

/// Ledger: "Py: loop blockers unnamed when return annotation
/// unreadable — bare `-> float` never judged; undetermined bodies
/// must name their blocker regardless." `samples` is a repetition-
/// window parameter (`repetition_window_element_pass`'s own shape),
/// so the loop runs through `stabilized_join`'s two-pass abstract
/// walk rather than a concrete per-element run.
///
/// The accumulation is spelled `total = total + s` — a plain
/// `Stmt::Assign`, not `total += s`. This matters: `loops.rs`'s
/// `AugAssign` arm calls the LOOP-LOCAL `binary_arithmetic_value`
/// directly (declines outright unless BOTH operands reduce to one
/// exact `Kind::Values` number), while `Stmt::Assign`'s RHS runs
/// through the ordinary `evaluate_expression` → `evaluate_binop` →
/// exact-arithmetic path: `total = total * 2.0` reads two exact
/// `Kind::Values` operands every iteration, so each judged pass
/// COMPLETES (`run_assign_once` accepts a `Kind::Values` result) and
/// the two passes bind DIFFERENT exact values ([2.0] then [4.0]) —
/// a write that provably never settles to a fixed point, reaching
/// `stabilized_join`'s havoc branch rather than declining the whole
/// loop before getting there (an element-consuming spelling like
/// `total + s` evaluates Values + Set to unknown inside the loop
/// module and declines the pass outright, landing on the older
/// "is not yet walked" blocker instead). A doubling rebind is
/// neither a `relational_sum` accumulation (`total += element`) nor
/// a count spelling (`count = count + 1`), so no more precise
/// recognizer intercepts it and the walk reaches
/// `walk_loop`/`stabilized_join` as an ordinary unrecognized rebind.
///
/// The return annotation is a BARE `float`, which `typereading::
/// declared_refinement` never reads (it only resolves a bare Name
/// through the module's own alias table), so `return_refinement` is
/// `None` and `walk_return` judges nothing on its own — before this
/// fix, that left the body with zero findings despite `total`'s
/// value never having been determined. The loop itself must still
/// name that blocker.
#[test]
fn a_for_loop_whose_accumulation_does_not_stabilize_names_its_blocker_even_under_a_bare_return_sort() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
        "def f(samples: Annotated[list[Sample], Field(min_length=1)]) -> float:\n",
        "    total = 1.0\n",
        "    for s in samples:\n",
        "        total = total * 2.0\n",
        "        pass\n",
        "    return total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(
        blockers.len(),
        1,
        "the loop's own non-stabilizing accumulation must be this body's named blocker, even though \
        the bare `-> float` return gives walk_return nothing to judge: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(blockers[0].message.contains("'total'"), "{}", blockers[0].message);
    assert!(
        blockers[0].message.contains("fixed point"),
        "the sentence must name WHY total is unreadable, not just that it is: {}",
        blockers[0].message
    );
}

/// The determined twin of the test above: a `for` loop over a
/// LITERAL list runs through the concrete per-element path
/// (`iterable_values`), never through `stabilized_join` at all, so
/// `widened_names` stays empty and the body — bare `-> float` return
/// included — records no blocker and no fire.
#[test]
fn a_for_loop_over_a_literal_list_stays_sentence_free_under_a_bare_return_sort() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def f() -> float:\n",
        "    total = 0.0\n",
        "    for s in [1.0, 2.0, 3.0]:\n",
        "        total += s\n",
        "    return total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "a concretely-executable for loop over a literal list must stay determined and sentence-free: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// showcase.py's own `invoice_total`/`refund_everything` shape: a
/// plain `total = total + amount` accumulation (never `total +=
/// amount`) over a `list[float]`-typed parameter — a repetition-window
/// element (`repetition_window_element_pass`'s own shape), so the
/// element `amount` is a `Kind::Set`, not one known number. This pins
/// whether `total + amount`'s arithmetic (`transfer_over_sets`,
/// `Values + Set` under `Add`) determines a sort-only answer that
/// `run_assign_once`/`bind_checked` can bind (walking the loop through
/// `stabilized_join`'s widen-to-unknown branch, the SAME non-
/// stabilizing-accumulation shape `a_for_loop_whose_accumulation_does_
/// not_stabilize_names_its_blocker_even_under_a_bare_return_sort`
/// already pins for a doubling rebind) — or declines the whole loop
/// outright, landing on the coarser "a for statement is not yet
/// walked" RTS7002 blocker instead.
#[test]
fn a_plain_rebind_accumulation_over_a_float_list_parameter_walks_the_loop() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Amounts = list[Annotated[float, Field(ge=0)]]\n",
        "type Total = Annotated[float, Field(ge=0)]\n",
        "def invoice_total(amounts: Amounts) -> Total:\n",
        "    total = 0.0\n",
        "    for amount in amounts:\n",
        "        total = total + amount\n",
        "    return total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let statement_blockers: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7002" && f.message.contains("is not yet walked"))
        .collect();
    assert!(
        statement_blockers.is_empty(),
        "the loop itself must be walked through stabilized_join, never left as the coarser \
        statement-shape blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// The designated-fire twin: the SAME accumulation shape, but the
/// running total only ever SUBTRACTS (`refund_everything`'s own row),
/// so the widened `total` still carries no lower bound past `0` at the
/// `-> Total` sink (`Total`'s own `Field(ge=0)`) — a returned value
/// this walk cannot prove satisfies the declared floor. Pins that once
/// the loop itself walks (this file's fix), the pre-existing
/// RETURN-THROUGH-LOOP CHANNEL / straight-line `walk_return` judging
/// still reaches its own designated fire on `total`'s value, rather
/// than the loop's own blocker swallowing it.
#[test]
fn a_plain_rebind_accumulation_that_only_subtracts_still_fires_its_return_sink() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Amounts = list[Annotated[float, Field(ge=0)]]\n",
        "type Total = Annotated[float, Field(ge=0)]\n",
        "def refund_everything(amounts: Amounts) -> Total:\n",
        "    total = 0.0\n",
        "    for amount in amounts:\n",
        "        total = total - amount\n",
        "    return total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let statement_blockers: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7002" && f.message.contains("is not yet walked"))
        .collect();
    assert!(
        statement_blockers.is_empty(),
        "the loop itself must be walked, never left as the coarser statement-shape blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}
