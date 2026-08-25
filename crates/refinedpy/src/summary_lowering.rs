//! A `def`'s body lowered ONCE into the kernel's flow IR, so the kernel
//! can compile it to a summary that every call site applies. The
//! interpreter in `summaries.rs` re-walks a callee's statements at every
//! call site with that call's own argument values; a summary quantifies
//! over all entries instead, so it is built from the DECLARATION alone
//! and nothing a call passes may influence what it lowers.
//!
//! The slot layout the compile and the apply both read:
//!
//! ```text
//!   [0 .. parameter_count)              one slot per declared parameter
//!   [parameter_count .. done_index)     one slot per name the body binds
//!   done_index                          the `#done` flag: {0} until a
//!                                       return raises it to {1}
//!   ret_index                           the `#ret` slot the returned
//!                                       value is written into
//! ```
//!
//! `done_index` and `ret_index` are the last two slots, matching the
//! kernel's own summary convention: the compiler's out-vector is the
//! whole binding row, so a slot's binding index and its out index are the
//! same number and a call site reads its answer at `ret_index`.
//!
//! TOTAL OR DECLINE. Every statement and every expression this file
//! reaches either lowers to exactly the IR the kernel walks, or the whole
//! `def` is UNLOWERABLE and the caller keeps the concrete interpreter.
//! Nothing lowers partially and nothing stands in for a construct that
//! was not read: a summary that admitted a havoc would serve a weaker
//! answer than the interpreter it is displacing.
//!
//! WHAT LOWERS, and what does not:
//!
//! | construct | lowering |
//! | --- | --- |
//! | a plain positional parameter (`posonlyargs`, `args`) | one entry slot |
//! | a parameter with a default, `*args`, `**kwargs`, a keyword-only parameter | declines |
//! | `name = <expr>` onto a bare name | `assign` into that name's slot |
//! | an assignment to a subscript, an attribute, a tuple/list target | declines |
//! | `name: T = <expr>` (`AnnAssign` with a value, bare-name target) | `assign` into that name's slot |
//! | `name += <expr>` / `-=` / `*=` (`AugAssign`, bare-name target) | `assign` of the binary effect |
//! | `return <expr>` | `assign` into `#ret`, then `#done := {1}` |
//! | a bare `return` | `#done := {1}`, leaving `#ret` at its absent entry |
//! | `if <test>: ... else: ...` on a lowerable test | `branch` carrying both arms |
//! | `pass` | no statement at all |
//! | a docstring (a bare string-literal statement) | no statement at all |
//! | every other statement (`for`, `while`, `match`, `try`, `raise`, `with`, a nested `def`/`class`, `global`/`nonlocal`, a bare expression statement) | declines |
//!
//! The EXPRESSION grammar, which both an assignment's right side and a
//! return's value read through:
//!
//! | expression | effect |
//! | --- | --- |
//! | a bare name naming a slot | `var` (a numeric read) |
//! | an `int`/`float` literal | `const` holding that one value |
//! | `True` / `False` | `const` holding 1 / 0 |
//! | `a * a` — the SAME source variable on both sides | `sq` (the correlated square image) |
//! | `a + b`, `a - b`, `a * b` | the matching `binary64` binary |
//! | `a / b`, `a // b`, `a % b`, `a ** b`, the bitwise operators | declines |
//! | `-a` | `binary64.neg` |
//! | `+a` | the operand's own effect |
//! | every other expression (a call, a subscript, an attribute, a string, a comparison, a comprehension, a container literal) | declines |
//!
//! The three arithmetic operators that lower are the ones whose Python
//! meaning and the kernel's `binary64` transfer agree on every operand:
//! addition, subtraction and multiplication of two numbers are the same
//! function in both. `/`, `//`, `%` and `**` are NOT — Python's `%` takes
//! the sign of its divisor where the kernel's `rem.truncDividendSign`
//! takes the dividend's, `//` floors, `/` answers a float from two ints,
//! and `**` has its own rows — so each declines rather than lowering onto
//! a transfer that means something else. This is the same line
//! `loops.rs::lower_counter_step_body` draws for a loop's step.
//!
//! The BRANCH tests that lower, all of them reading ONE slot:
//!
//! | test | lowering |
//! | --- | --- |
//! | `name` (a bare name, truthiness) | `js.truthyNum` on that slot |
//! | `name < literal`, `<=`, `>`, `>=`, `==` | the matching one-slot comparison, the literal in `w` |
//! | `name < other`, `<=`, `>`, `>=`, `==` between two slots | the matching `*Slot` comparison, the second slot in `on_b` |
//! | `name is None` / `name is not None` | `js.eqNull` (the arms swap for `is not`) |
//! | every other test (`and`/`or`/`not`, a chained comparison, a call, a string literal operand) | declines |

use ruff_python_ast::CmpOp;
use ruff_python_ast::ElifElseClause;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtIf;
use ruff_python_ast::UnaryOp;
use refined_kernel::loop_questions::IrBranchTest;
use refined_kernel::loop_questions::IrStatement;
use refined_kernel::loop_questions::IrStatementKind;
use refined_kernel::loop_questions::LoopEffect;
use refined_kernel::loop_questions::LoopEffectKind;
use refined_kernel::loop_questions::LoopEffectOp;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;

/// A `def`'s body as the kernel's flow IR, plus the slot bookkeeping a
/// call site needs to build entry states and read the answer back out.
#[derive(Debug, Clone)]
pub struct LoweredBody {
    /// The body's statements, in order.
    pub statements: Vec<IrStatement>,
    /// How many leading slots are PARAMETERS — the entries a call site
    /// fills from its own arguments. Every slot past this one enters
    /// absent, except the done flag, which enters at {0}.
    pub parameter_count: usize,
    /// Every slot, locals and the two bookkeeping slots included. The
    /// compile is asked for this arity, and the apply sends this many
    /// entry states.
    pub slot_count: usize,
    /// Where the `#done` flag sits.
    pub done_index: usize,
    /// Where the returned value sits — the out index a call site reads.
    pub ret_index: usize,
}

/// `def`'s body lowered, or `None` when any part of it falls outside the
/// grammar above. A `None` is PERMANENT for that `def`: nothing about a
/// call site can make a body lower that did not, so the registry
/// (`summaries.rs`) remembers the refusal rather than paying the walk at
/// every call.
pub fn lower_function_body(def: &StmtFunctionDef) -> Option<LoweredBody> {
    let layout = slot_layout(def)?;
    let lowering = Lowering {
        slots: layout.slots,
        done_index: layout.done_index,
        ret_index: layout.ret_index,
    };
    let statements = lowering.lower_statements(&def.body)?;
    Some(LoweredBody {
        statements,
        parameter_count: layout.parameter_count,
        // the result slot is the LAST one, so its index plus one is the
        // whole count
        slot_count: layout.ret_index + 1,
        done_index: layout.done_index,
        ret_index: layout.ret_index,
    })
}

/// The slot names in index order, then the two bookkeeping indices past
/// them. Built before any statement lowers so a forward read of a name
/// the body binds later still resolves to the slot it will be written
/// into — the kernel's own entry state for that slot is absent, which is
/// exactly what an unwritten local holds.
struct SlotLayout {
    slots: Vec<String>,
    parameter_count: usize,
    done_index: usize,
    ret_index: usize,
}

/// The parameters, then every name the body binds, then the flag and the
/// result slot. `None` where the parameter list carries a shape a summary
/// cannot quantify over: a default (whose value would have to be
/// evaluated against a scope this lowering does not hold), a `*args` or
/// `**kwargs` tail (whose entry has no one state), or a keyword-only
/// parameter (which a positional entry vector cannot place).
fn slot_layout(def: &StmtFunctionDef) -> Option<SlotLayout> {
    if def.parameters.vararg.is_some() || def.parameters.kwarg.is_some() {
        return None;
    }
    if !def.parameters.kwonlyargs.is_empty() {
        return None;
    }
    let mut slots: Vec<String> = Vec::new();
    for parameter in def.parameters.posonlyargs.iter().chain(def.parameters.args.iter()) {
        if parameter.default.is_some() {
            return None;
        }
        slots.push(parameter.parameter.name.id.as_str().to_owned());
    }
    let parameter_count = slots.len();
    collect_bound_names(&def.body, &mut slots);
    let done_index = slots.len();
    let ret_index = done_index + 1;
    Some(SlotLayout { slots, parameter_count, done_index, ret_index })
}

/// Every bare name the body assigns to, appended once each in the order
/// they first appear. Only the statement forms this file lowers are
/// walked — a name bound by a form that declines never reaches a slot,
/// because the statement carrying it declines the whole body first.
fn collect_bound_names(body: &[Stmt], slots: &mut Vec<String>) {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => {
                if let [Expr::Name(name)] = assign.targets.as_slice() {
                    push_slot(slots, name.id.as_str());
                }
            }
            Stmt::AnnAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    push_slot(slots, name.id.as_str());
                }
            }
            Stmt::AugAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    push_slot(slots, name.id.as_str());
                }
            }
            Stmt::If(if_stmt) => {
                collect_bound_names(&if_stmt.body, slots);
                for clause in &if_stmt.elif_else_clauses {
                    collect_bound_names(&clause.body, slots);
                }
            }
            _ => {}
        }
    }
}

fn push_slot(slots: &mut Vec<String>, name: &str) {
    if !slots.iter().any(|held| held == name) {
        slots.push(name.to_owned());
    }
}

/// The lowering's own state: which name sits at which slot, and where the
/// two bookkeeping slots are.
struct Lowering {
    slots: Vec<String>,
    done_index: usize,
    ret_index: usize,
}

impl Lowering {
    /// The slot a name reads or writes, or `None` for a name this body
    /// never binds — a module-level global, a builtin, an enclosing
    /// scope's capture. A summary quantifies over its own entries alone,
    /// so a name outside them has no entry to stand for and the body
    /// declines rather than reading whatever a call site happens to hold.
    fn slot_of(&self, name: &str) -> Option<i64> {
        self.slots.iter().position(|held| held == name).map(|index| index as i64)
    }

    fn lower_statements(&self, body: &[Stmt]) -> Option<Vec<IrStatement>> {
        let mut out: Vec<IrStatement> = Vec::new();
        for (index, stmt) in body.iter().enumerate() {
            match stmt {
                Stmt::Pass(_) => {}
                // a docstring is documentation, never a readable effect —
                // it binds nothing and computes nothing, so it lowers to no
                // statement at all rather than declining the body
                Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::StringLiteral(_)) => {}
                Stmt::Assign(assign) => {
                    let [Expr::Name(name)] = assign.targets.as_slice() else {
                        return None;
                    };
                    let target = self.slot_of(name.id.as_str())?;
                    let effect = self.lower_expression(assign.value.as_ref())?;
                    out.push(assign_statement(target, effect));
                }
                Stmt::AnnAssign(assign) => {
                    let Expr::Name(name) = assign.target.as_ref() else {
                        return None;
                    };
                    // a bare `name: T` with no value BINDS NOTHING at
                    // runtime (it only records an annotation), so a body
                    // carrying one would have a slot whose entry state and
                    // whose post-statement state disagree about whether the
                    // name exists — declined rather than lowered as a write
                    let value = assign.value.as_deref()?;
                    let target = self.slot_of(name.id.as_str())?;
                    let effect = self.lower_expression(value)?;
                    out.push(assign_statement(target, effect));
                }
                Stmt::AugAssign(assign) => {
                    let Expr::Name(name) = assign.target.as_ref() else {
                        return None;
                    };
                    let target = self.slot_of(name.id.as_str())?;
                    let op = arithmetic_op(assign.op)?;
                    let left = LoopEffect {
                        kind: LoopEffectKind::Var,
                        index: target,
                        ..Default::default()
                    };
                    let right = self.lower_expression(assign.value.as_ref())?;
                    out.push(assign_statement(target, binary_effect(op, left, right)));
                }
                Stmt::Return(ret) => {
                    // the result slot takes the value, the flag rises, and
                    // the rest of this block never runs — dead statements
                    // simply do not lower. A bare `return` raises the flag
                    // alone, leaving `#ret` at the absent entry state, which
                    // IS the `None` a bare return produces.
                    if let Some(value) = ret.value.as_deref() {
                        let effect = self.lower_expression(value)?;
                        out.push(assign_statement(self.ret_index as i64, as_var_state(effect)));
                    }
                    out.push(raise_done(self.done_index as i64));
                    return Some(out);
                }
                Stmt::If(if_stmt) => {
                    let branch = self.lower_if(if_stmt)?;
                    let arm_returned = raises_done(std::slice::from_ref(&branch), self.done_index as i64);
                    out.push(branch);
                    // AN ARM THAT MAY HAVE RETURNED. The kernel's branch
                    // joins both arms and the walk carries on, so the
                    // block's REMAINDER would otherwise run on the
                    // returning path too — and its own `return` would
                    // overwrite a result slot the earlier arm had already
                    // filled. The remainder therefore runs only where the
                    // done flag stayed DOWN, as an ordinary branch on the
                    // flag: the then side (flag up, already returned) is
                    // empty, the else side carries the rest.
                    if arm_returned {
                        let rest = self.lower_statements(&body[index + 1..])?;
                        if !rest.is_empty() {
                            out.push(IrStatement {
                                kind: IrStatementKind::Branch,
                                on: self.done_index as i64,
                                test: IrBranchTest::TruthyNum,
                                then: Vec::new(),
                                else_: rest,
                                ..Default::default()
                            });
                        }
                        return Some(out);
                    }
                }
                _ => return None,
            }
        }
        Some(out)
    }

    /// `if <test>: <then> [elif ...] [else: <else>]` as one branch
    /// statement. An `elif` chain nests: the chain's tail lowers as the
    /// branch's else arm, exactly the shape the source spells.
    fn lower_if(&self, if_stmt: &StmtIf) -> Option<IrStatement> {
        let (on, test, w, on_b) = self.lower_test(if_stmt.test.as_ref())?;
        let then_body = self.lower_statements(&if_stmt.body)?;
        let else_body = self.lower_else_clauses(&if_stmt.elif_else_clauses)?;
        Some(IrStatement {
            kind: IrStatementKind::Branch,
            on,
            test,
            w,
            on_b,
            then: then_body,
            else_: else_body,
            ..Default::default()
        })
    }

    /// The statements an `if`'s else side runs. A bare `else:` is its own
    /// body; an `elif` is one nested branch statement whose own else side
    /// is the rest of the chain.
    fn lower_else_clauses(&self, clauses: &[ElifElseClause]) -> Option<Vec<IrStatement>> {
        let Some((first, rest)) = clauses.split_first() else {
            return Some(Vec::new());
        };
        let Some(test) = first.test.as_ref() else {
            // a bare `else:` ends the chain — anything after it is not
            // reachable syntax, so a clause list continuing past one is a
            // shape this lowering does not spell
            if !rest.is_empty() {
                return None;
            }
            return self.lower_statements(&first.body);
        };
        let (on, branch_test, w, on_b) = self.lower_test(test)?;
        let then_body = self.lower_statements(&first.body)?;
        let else_body = self.lower_else_clauses(rest)?;
        Some(vec![IrStatement {
            kind: IrStatementKind::Branch,
            on,
            test: branch_test,
            w,
            on_b,
            then: then_body,
            else_: else_body,
            ..Default::default()
        }])
    }

    /// A branch test as the slot it reads, the test name, and whichever
    /// operand the test carries — a constant in `w`, or a second slot in
    /// `on_b`. Every test reads ONE slot on its left, which is what the
    /// kernel's branch statement carries; a test over anything else
    /// declines the body.
    fn lower_test(&self, test: &Expr) -> Option<(i64, IrBranchTest, Option<f64>, i64)> {
        if let Expr::Name(name) = test {
            // a bare name in test position is Python's truthiness. It
            // lowers only as the NUMERIC truthiness the kernel decides:
            // a string, a list or a dict in this position has its own
            // emptiness rule the numeric test does not spell.
            let slot = self.slot_of(name.id.as_str())?;
            return Some((slot, IrBranchTest::TruthyNum, None, 0));
        }
        let Expr::Compare(compare) = test else {
            return None;
        };
        // a CHAINED comparison (`0 < x < 10`) is two tests and one
        // short-circuit, not one branch — declined rather than read as
        // its first link
        let ([op], [right]) = (compare.ops.as_ref(), compare.comparators.as_ref()) else {
            return None;
        };
        let op = *op;
        let Expr::Name(left) = compare.left.as_ref() else {
            return None;
        };
        let slot = self.slot_of(left.id.as_str())?;
        // `x is None`: the kernel's own null test, which decides exactly
        // the null admission and leaves the undefined one untouched —
        // the same correspondence `entry_state_of` builds when it puts a
        // `Kind::Null` argument's admission on the wire.
        //
        // `x is not None` is that test with its two ARMS exchanged, which
        // this function has no way to say: it answers a test, not a pair
        // of arms, and the arms are lowered by its caller. It declines,
        // and so does any `is` against something other than `None` —
        // identity against an object is not a set question at all.
        if matches!(op, CmpOp::Is) && matches!(right, Expr::NoneLiteral(_)) {
            return Some((slot, IrBranchTest::EqNull, None, 0));
        }
        if matches!(op, CmpOp::Is | CmpOp::IsNot) {
            return None;
        }
        if let Expr::Name(other) = right {
            let other_slot = self.slot_of(other.id.as_str())?;
            let test = match op {
                CmpOp::Lt => IrBranchTest::LtSlot,
                CmpOp::LtE => IrBranchTest::LeSlot,
                CmpOp::Gt => IrBranchTest::GtSlot,
                CmpOp::GtE => IrBranchTest::GeSlot,
                CmpOp::Eq => IrBranchTest::EqSlot,
                _ => return None,
            };
            return Some((slot, test, None, other_slot));
        }
        let value = number_literal(right)?;
        let test = match op {
            CmpOp::Lt => IrBranchTest::Lt,
            CmpOp::LtE => IrBranchTest::Le,
            CmpOp::Gt => IrBranchTest::Gt,
            CmpOp::GtE => IrBranchTest::Ge,
            CmpOp::Eq => IrBranchTest::Eq,
            _ => return None,
        };
        Some((slot, test, Some(value), 0))
    }

    /// An expression as one effect over this body's slots, or `None` for
    /// anything outside the grammar in this module's doc.
    fn lower_expression(&self, expression: &Expr) -> Option<LoopEffect> {
        match expression {
            Expr::Name(name) => {
                let index = self.slot_of(name.id.as_str())?;
                Some(LoopEffect {
                    kind: LoopEffectKind::Var,
                    index,
                    ..Default::default()
                })
            }
            Expr::NumberLiteral(_) | Expr::BooleanLiteral(_) => {
                let value = number_literal(expression)?;
                Some(constant_effect(value))
            }
            Expr::UnaryOp(unary) => match unary.op {
                UnaryOp::USub => {
                    let operand = self.lower_expression(unary.operand.as_ref())?;
                    Some(LoopEffect {
                        kind: LoopEffectKind::Unary,
                        op: LoopEffectOp::Neg,
                        a: Some(Box::new(operand)),
                        ..Default::default()
                    })
                }
                // unary `+` on a number is the identity (CPython's own
                // `int.__pos__`/`float.__pos__` answer the operand), so it
                // lowers to the operand's own effect
                UnaryOp::UAdd => self.lower_expression(unary.operand.as_ref()),
                _ => None,
            },
            // `x * x` — the SAME source variable on both sides — is a
            // structural square: the kernel's `Effect.sq` answers the
            // correlated image `[0, max²]`, which the general product
            // `Binary(Mul, Var(i), Var(i))` cannot, since the kernel no
            // longer recognizes x*x by syntax (unsound under
            // renaming). Read directly off the two `Expr::Name` nodes,
            // before either side lowers: this is the one place the
            // identifier binding is honestly known.
            Expr::BinOp(binop) if is_same_name_square(binop) => {
                let Expr::Name(name) = binop.left.as_ref() else {
                    unreachable!("is_same_name_square only matches two Expr::Name operands");
                };
                let index = self.slot_of(name.id.as_str())?;
                Some(LoopEffect {
                    kind: LoopEffectKind::Sq,
                    index,
                    ..Default::default()
                })
            }
            Expr::BinOp(binop) => {
                let op = arithmetic_op(binop.op)?;
                let left = self.lower_expression(binop.left.as_ref())?;
                let right = self.lower_expression(binop.right.as_ref())?;
                Some(binary_effect(op, left, right))
            }
            _ => None,
        }
    }
}

/// Whether a `BinOp` is `<name> * <name>` for the SAME identifier on
/// both sides — decided from the source AST's own two `Expr::Name`
/// nodes, never from a lowered effect, which has already erased which
/// variable a term came from. A different identifier on each side (or
/// anything but two bare names) keeps the general `Mul` lowering,
/// since only a proven-identical source binding licenses the
/// correlated square.
fn is_same_name_square(binop: &ruff_python_ast::ExprBinOp) -> bool {
    if !matches!(binop.op, Operator::Mult) {
        return false;
    }
    let (Expr::Name(left), Expr::Name(right)) = (binop.left.as_ref(), binop.right.as_ref()) else {
        return false;
    };
    left.id.as_str() == right.id.as_str()
}

/// The three Python operators whose meaning the kernel's `binary64`
/// transfers state exactly. `/`, `//`, `%` and `**` are absent
/// deliberately: each names a function the kernel's own transfer computes
/// differently for Python's operands (this module's doc holds the
/// argument), so each declines the body rather than lowering onto a
/// transfer that means something else.
fn arithmetic_op(op: Operator) -> Option<LoopEffectOp> {
    match op {
        Operator::Add => Some(LoopEffectOp::Add),
        Operator::Sub => Some(LoopEffectOp::Sub),
        Operator::Mult => Some(LoopEffectOp::Mul),
        _ => None,
    }
}

/// The one number a literal spells: an `int` or `float` literal's own
/// value, or a bool's 1/0 (Python's `bool` is an `int` subclass, so `True`
/// in an arithmetic position IS 1). A unary `-`/`+` wrapping a literal is
/// read through, the same reading `loops.rs::number_literal_value`
/// performs. A complex literal, and an int too large for i64, have no
/// exact place on the real line and answer `None`.
fn number_literal(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(value) => value.as_i64().map(|whole| whole as f64),
            Number::Float(value) => Some(*value),
            Number::Complex { .. } => None,
        },
        Expr::BooleanLiteral(literal) => Some(if literal.value { 1.0 } else { 0.0 }),
        Expr::UnaryOp(unary) => {
            let operand = number_literal(unary.operand.as_ref())?;
            match unary.op {
                UnaryOp::USub => Some(-operand),
                UnaryOp::UAdd => Some(operand),
                _ => None,
            }
        }
        _ => None,
    }
}

fn constant_effect(value: f64) -> LoopEffect {
    LoopEffect {
        kind: LoopEffectKind::Const,
        set: make_refined_set(vec![one_of(&[value])]),
        ..Default::default()
    }
}

fn binary_effect(op: LoopEffectOp, left: LoopEffect, right: LoopEffect) -> LoopEffect {
    LoopEffect {
        kind: LoopEffectKind::Binary,
        op,
        a: Some(Box::new(left)),
        b: Some(Box::new(right)),
        ..Default::default()
    }
}

/// A bare slot read written into the RESULT slot copies the source's WHOLE
/// state — its set, and its absent and NaN admissions — rather than the
/// numeric read `Var` performs, which would coerce an absent source to
/// NaN. `return x` hands back exactly what `x` held.
fn as_var_state(effect: LoopEffect) -> LoopEffect {
    if matches!(effect.kind, LoopEffectKind::Var) {
        return LoopEffect {
            kind: LoopEffectKind::VarState,
            index: effect.index,
            ..Default::default()
        };
    }
    effect
}

/// Whether a lowered statement list writes the done flag anywhere — the
/// test that tells a RETURNING arm from a straight one, which is what
/// decides whether the block's remainder needs its own gate on the flag.
/// Branch arms are walked, since a `return` inside either one raises the
/// flag exactly as a top-level one does.
fn raises_done(statements: &[IrStatement], done_index: i64) -> bool {
    statements.iter().any(|statement| match statement.kind {
        IrStatementKind::Assign => statement.target == done_index,
        IrStatementKind::Branch | IrStatementKind::BranchBoth => {
            raises_done(&statement.then, done_index) || raises_done(&statement.else_, done_index)
        }
        _ => false,
    })
}

fn assign_statement(target: i64, effect: LoopEffect) -> IrStatement {
    IrStatement {
        kind: IrStatementKind::Assign,
        target,
        effect,
        ..Default::default()
    }
}

/// `#done := {1}` — the statement every `return` ends with, whatever it
/// wrote into the result slot.
fn raise_done(done_index: i64) -> IrStatement {
    assign_statement(done_index, constant_effect(1.0))
}

#[cfg(test)]
mod tests {
    use refined_kernel::loop_questions::stmt_wire;
    use ruff_python_parser::parse_module;

    use super::*;

    /// Parses `source` as a module and returns its single top-level `def`.
    fn parsed_def(source: &str) -> StmtFunctionDef {
        let module = parse_module(source).expect("fixture source parses").into_syntax();
        let stmt = module.body.into_iter().next().expect("one top-level statement");
        stmt.function_def_stmt().expect("top-level statement is a def")
    }

    /// The lowered statements as the JSON the kernel reads them from —
    /// the one spelling both sides agree on, so a shape test compares
    /// exactly what crosses.
    fn wires(lowered: &LoweredBody) -> Vec<String> {
        lowered.statements.iter().map(stmt_wire).collect()
    }

    #[test]
    fn a_two_statement_body_lowers_to_an_assign_a_result_write_and_the_flag() {
        let def = parsed_def("def scale(x):\n    doubled = x + x\n    return doubled\n");
        let lowered = lower_function_body(&def).expect("straight-line arithmetic lowers");
        // slots: x=0, doubled=1, #done=2, #ret=3
        assert_eq!(lowered.parameter_count, 1);
        assert_eq!(lowered.slot_count, 4);
        assert_eq!(lowered.done_index, 2);
        assert_eq!(lowered.ret_index, 3);
        assert_eq!(
            wires(&lowered),
            vec![
                r#"{"assign":{"target":1,"e":{"op":"binary64.add","A":{"var":0},"B":{"var":0}}}}"#,
                r#"{"assign":{"target":3,"e":{"varState":1}}}"#,
                r#"{"assign":{"target":2,"e":{"set":{"forms":[{"form":"oneOf","w":[{"num":1,"exp":0}]}]}}}}"#,
            ],
        );
    }

    #[test]
    fn a_literal_return_writes_the_constant_into_the_result_slot() {
        let def = parsed_def("def one():\n    return 1\n");
        let lowered = lower_function_body(&def).expect("a literal return lowers");
        assert_eq!(lowered.parameter_count, 0);
        assert_eq!(lowered.slot_count, 2);
        assert_eq!(
            wires(&lowered),
            vec![
                r#"{"assign":{"target":1,"e":{"set":{"forms":[{"form":"oneOf","w":[{"num":1,"exp":0}]}]}}}}"#,
                r#"{"assign":{"target":0,"e":{"set":{"forms":[{"form":"oneOf","w":[{"num":1,"exp":0}]}]}}}}"#,
            ],
        );
    }

    #[test]
    fn a_bare_return_raises_the_flag_and_writes_no_result() {
        let def = parsed_def("def nothing():\n    return\n");
        let lowered = lower_function_body(&def).expect("a bare return lowers");
        assert_eq!(lowered.statements.len(), 1);
        assert_eq!(lowered.statements[0].target, lowered.done_index as i64);
    }

    #[test]
    fn an_if_else_lowers_to_one_branch_carrying_both_arms() {
        let def = parsed_def("def pick(flag):\n    if flag:\n        return 3\n    else:\n        return 5\n");
        let lowered = lower_function_body(&def).expect("a branch on a slot's truthiness lowers");
        let [branch] = lowered.statements.as_slice() else {
            panic!("want exactly one branch statement, got {:?}", lowered.statements);
        };
        assert!(matches!(branch.kind, IrStatementKind::Branch));
        assert!(matches!(branch.test, IrBranchTest::TruthyNum));
        assert_eq!(branch.on, 0);
        assert_eq!(branch.then.len(), 2, "the then arm writes the result then raises the flag");
        assert_eq!(branch.else_.len(), 2, "the else arm writes the result then raises the flag");
    }

    #[test]
    fn an_elif_chain_nests_as_the_branch_statements_else_arm() {
        let def = parsed_def(
            "def band(n):\n    if n < 10:\n        return 1\n    elif n < 20:\n        return 2\n    else:\n        return 3\n",
        );
        let lowered = lower_function_body(&def).expect("an elif chain lowers");
        let [outer] = lowered.statements.as_slice() else {
            panic!("want one outer branch, got {:?}", lowered.statements);
        };
        assert!(matches!(outer.test, IrBranchTest::Lt));
        assert_eq!(outer.w, Some(10.0));
        let [inner] = outer.else_.as_slice() else {
            panic!("want the elif nested in the else arm, got {:?}", outer.else_);
        };
        assert!(matches!(inner.test, IrBranchTest::Lt));
        assert_eq!(inner.w, Some(20.0));
    }

    /// A guard whose arm returns, followed by more statements: the
    /// remainder must run only where the flag stayed DOWN, or the later
    /// `return` would overwrite the result the guard's arm already wrote.
    #[test]
    fn statements_after_a_returning_guard_are_gated_on_the_done_flag() {
        let def = parsed_def("def guarded(n):\n    if n < 0:\n        return 0\n    return n\n");
        let lowered = lower_function_body(&def).expect("a guard then a return lowers");
        let [guard, gate] = lowered.statements.as_slice() else {
            panic!("want the guard and its continuation gate, got {:?}", lowered.statements);
        };
        assert!(matches!(guard.test, IrBranchTest::Lt));
        assert!(matches!(gate.kind, IrStatementKind::Branch));
        assert_eq!(gate.on, lowered.done_index as i64, "the gate reads the done flag");
        assert!(gate.then.is_empty(), "the flag-up path already returned");
        assert_eq!(gate.else_.len(), 2, "the remainder writes the result then raises the flag");
    }

    /// A guard whose arms do NOT return needs no gate: the branch joins
    /// and the block carries straight on.
    #[test]
    fn statements_after_a_non_returning_guard_carry_on_ungated() {
        let def = parsed_def("def adjusted(n):\n    if n < 0:\n        n = 0\n    return n\n");
        let lowered = lower_function_body(&def).expect("a non-returning guard lowers");
        assert_eq!(lowered.statements.len(), 3, "the branch, the result write, the flag: {:?}", lowered.statements);
        assert!(matches!(lowered.statements[0].kind, IrStatementKind::Branch));
        assert!(matches!(lowered.statements[1].kind, IrStatementKind::Assign));
    }

    #[test]
    fn a_comparison_between_two_slots_lowers_to_the_two_slot_test() {
        let def = parsed_def("def smaller(a, b):\n    if a < b:\n        return a\n    return b\n");
        let lowered = lower_function_body(&def).expect("a two-slot comparison lowers");
        let branch = &lowered.statements[0];
        assert!(matches!(branch.test, IrBranchTest::LtSlot));
        assert_eq!(branch.on, 0);
        assert_eq!(branch.on_b, 1);
    }

    #[test]
    fn an_augmented_assignment_lowers_as_the_binary_over_its_own_slot() {
        let def = parsed_def("def bump(n):\n    n += 1\n    return n\n");
        let lowered = lower_function_body(&def).expect("an augmented assignment lowers");
        assert_eq!(
            wires(&lowered)[0],
            r#"{"assign":{"target":0,"e":{"op":"binary64.add","A":{"var":0},"B":{"set":{"forms":[{"form":"oneOf","w":[{"num":1,"exp":0}]}]}}}}}"#,
        );
    }

    #[test]
    fn a_docstring_and_a_pass_lower_to_no_statements_of_their_own() {
        let def = parsed_def("def documented(x):\n    \"the doc\"\n    pass\n    return x\n");
        let lowered = lower_function_body(&def).expect("a docstring and a pass lower");
        assert_eq!(lowered.statements.len(), 2, "only the result write and the flag: {:?}", lowered.statements);
    }

    #[test]
    fn division_declines_the_whole_body() {
        // Python's `/` answers a float from two ints and its `//` floors;
        // the kernel's binary64.div is neither, so the body declines
        // rather than lowering onto a transfer meaning something else.
        assert!(lower_function_body(&parsed_def("def half(x):\n    return x / 2\n")).is_none());
        assert!(lower_function_body(&parsed_def("def half(x):\n    return x // 2\n")).is_none());
        assert!(lower_function_body(&parsed_def("def rest(x):\n    return x % 2\n")).is_none());
        assert!(lower_function_body(&parsed_def("def square(x):\n    return x ** 2\n")).is_none());
    }

    #[test]
    fn a_name_multiplied_by_itself_lowers_to_the_structural_square() {
        // slots: x=0, #done=1, #ret=2 — the return writes #ret first
        let def = parsed_def("def square(x):\n    return x * x\n");
        let lowered = lower_function_body(&def).expect("x * x lowers");
        assert_eq!(
            wires(&lowered)[0],
            r#"{"assign":{"target":2,"e":{"sq":0}}}"#,
            "x * x must lower to the sq effect, not the general mul"
        );
    }

    #[test]
    fn two_distinct_names_multiplied_keep_the_general_mul_lowering() {
        // slots: x=0, y=1, #done=2, #ret=3
        let def = parsed_def("def product(x, y):\n    return x * y\n");
        let lowered = lower_function_body(&def).expect("x * y lowers");
        assert_eq!(
            wires(&lowered)[0],
            r#"{"assign":{"target":3,"e":{"op":"binary64.mul","A":{"var":0},"B":{"var":1}}}}"#,
            "distinct operands must keep the general mul lowering"
        );
    }

    #[test]
    fn a_name_the_body_never_binds_declines() {
        // `LIMIT` is a module-level global: a summary quantifies over its
        // own entries alone, so there is no entry standing for it.
        assert!(lower_function_body(&parsed_def("def capped(x):\n    return x + LIMIT\n")).is_none());
    }

    #[test]
    fn a_parameter_shape_a_positional_entry_vector_cannot_place_declines() {
        assert!(lower_function_body(&parsed_def("def defaulted(x, y=1):\n    return x + y\n")).is_none());
        assert!(lower_function_body(&parsed_def("def varargs(*xs):\n    return 1\n")).is_none());
        assert!(lower_function_body(&parsed_def("def kwargs(**rest):\n    return 1\n")).is_none());
        assert!(lower_function_body(&parsed_def("def kwonly(x, *, y):\n    return x + y\n")).is_none());
    }

    #[test]
    fn a_statement_outside_the_grammar_declines_the_whole_body() {
        assert!(lower_function_body(&parsed_def("def looped(x):\n    for i in [1]:\n        x = x + i\n    return x\n")).is_none());
        assert!(lower_function_body(&parsed_def("def raised(x):\n    raise ValueError\n")).is_none());
        assert!(lower_function_body(&parsed_def("def called(x):\n    return f(x)\n")).is_none());
        assert!(lower_function_body(&parsed_def("def indexed(x):\n    return x[0]\n")).is_none());
        assert!(lower_function_body(&parsed_def("def attributed(x):\n    return x.age\n")).is_none());
    }

    #[test]
    fn a_test_the_branch_grammar_does_not_carry_declines() {
        assert!(lower_function_body(&parsed_def("def both(a, b):\n    if a and b:\n        return 1\n    return 0\n")).is_none());
        assert!(lower_function_body(&parsed_def("def chained(a):\n    if 0 < a < 9:\n        return 1\n    return 0\n")).is_none());
        assert!(lower_function_body(&parsed_def("def negated(a):\n    if not a:\n        return 1\n    return 0\n")).is_none());
        assert!(lower_function_body(&parsed_def("def present(a):\n    if a is not None:\n        return 1\n    return 0\n")).is_none());
    }

    #[test]
    fn an_is_none_test_lowers_to_the_kernels_null_test() {
        let def = parsed_def("def absent(a):\n    if a is None:\n        return 0\n    return 1\n");
        let lowered = lower_function_body(&def).expect("an `is None` test lowers");
        assert!(matches!(lowered.statements[0].test, IrBranchTest::EqNull));
    }
}
