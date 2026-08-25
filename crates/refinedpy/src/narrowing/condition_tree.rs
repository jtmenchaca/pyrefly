//! SET channel: condition → `NarrowTree`, plus `meet_set_answer`.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::trust_grades::trust_level_of;
use refined_kernel::kernel_interface::NarrowTree;
use refined_kernel::narrow_questions::NarrowCmpOp;
use refined_kernel::narrow_questions::NarrowTreeKind;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::UnaryOp;

use super::compare::NumericCmpOp;
use super::compare::mirror_cmp_op;
use super::compare::numeric_cmp_op;
use super::literal_number;
use super::name_of;

// ── the SET channel: condition → NarrowTree ─────────────────────────

/// A `NarrowTree` leaf that claims nothing — the kernel's own "no
/// reading" leaf (`gate_narrow`/`narrow_wire`'s `Other` arm never reads
/// its other fields), matching `refined-ts-go/internal/refinedts/
/// narrowing/type_guard_recognizers.go`'s package-level `Other` value.
pub(super) fn other_tree() -> NarrowTree {
    NarrowTree {
        kind: NarrowTreeKind::Other,
        op: None,
        k: 0.0,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: None,
        b: None,
    }
}

/// One `Cmp` leaf (`name op literal`) — the bare-fields constructor
/// every other `NarrowTree` variant this file never builds also needs,
/// since the struct derives no `Default` (`refined_kernel::
/// narrow_questions` — every field explicit at every call site, the
/// same discipline that module's own tests follow).
pub(super) fn cmp_tree(op: NarrowCmpOp, k: f64) -> NarrowTree {
    NarrowTree {
        kind: NarrowTreeKind::Cmp,
        op: Some(op),
        k,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: None,
        b: None,
    }
}

pub(super) fn not_tree(a: NarrowTree) -> NarrowTree {
    NarrowTree {
        kind: NarrowTreeKind::Not,
        op: None,
        k: 0.0,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: Some(Box::new(a)),
        b: None,
    }
}

pub(super) fn and_or_tree(kind: NarrowTreeKind, a: NarrowTree, b: NarrowTree) -> NarrowTree {
    NarrowTree {
        kind,
        op: None,
        k: 0.0,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: Some(Box::new(a)),
        b: Some(Box::new(b)),
    }
}

/// `NumericCmpOp` → the kernel's own `NarrowCmpOp` — `Eq`/`NotEq` are
/// NOT `Cmp` leaves (the kernel's `Cmp` carries only the four ORDER
/// operators; equality is its own `NarrowTreeKind::Eq`/its negation),
/// so this reads only the four this file's `Cmp` leaf can name; `None`
/// for `Eq`/`NotEq`, read directly by `condition_tree_of` instead.
pub(super) fn narrow_cmp_op_of(op: NumericCmpOp) -> Option<NarrowCmpOp> {
    match op {
        NumericCmpOp::Lt => Some(NarrowCmpOp::Lt),
        NumericCmpOp::LtE => Some(NarrowCmpOp::Le),
        NumericCmpOp::Gt => Some(NarrowCmpOp::Gt),
        NumericCmpOp::GtE => Some(NarrowCmpOp::Ge),
        NumericCmpOp::Eq | NumericCmpOp::NotEq => None,
    }
}

/// Lowers `condition`, RELATIVE TO `place` (the one name the kernel's
/// narrowing question is always scoped to — `refined-ts-go`'s own
/// `TreeOf`/`leafTreeOf` convention), into the kernel's `NarrowTree`
/// grammar: `!`/`and`/`or` fold the same shape/structure the condition
/// itself has (a `not` node wraps its operand's own tree UNCHANGED —
/// the KERNEL's own `narrowQ` is what swaps a `not`'s when-true/
/// when-false pair, `set_functions/narrow.lean`'s `.not t => (p.2,
/// p.1)`, so this builder never needs to track which polarity it is
/// "under" the way the VALUES channel's `narrow`/`narrow_bool_op` does
/// — the caller (`narrow_set_kind_names`) reads whichever side of the
/// ONE resulting `NarrowAnswer` its own branch truth names).
///
/// Any leaf NOT on `place`, or NOT one of this file's recognized
/// shapes (a call other than `isinstance`, a string test, two changing
/// names, `is`/`is not None` — Python's `None` is never a member of a
/// `Kind::Set`'s own domain, so an absence test states nothing a set
/// claim could narrow), lowers to `other_tree()` — the honest "no
/// claim" leaf, never a guess. The WHOLE tree is `None` only when
/// `condition` itself has no shape this function reads at all (an
/// unreachable case today — every `Expr` arm below returns `Some`,
/// down to the catch-all `other_tree()` — kept `Option` so a future
/// leaf that must genuinely decline has the same "no tree at all" exit
/// `narrow_set_kind_names`'s `let Some(tree) = … else { continue }`
/// already expects).
pub(super) fn condition_tree_of(condition: &Expr, place: &str) -> Option<NarrowTree> {
    match condition {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => {
            condition_tree_of(&unary.operand, place).map(not_tree)
        }
        Expr::BoolOp(bool_op) => {
            let kind = match bool_op.op {
                BoolOp::And => NarrowTreeKind::And,
                BoolOp::Or => NarrowTreeKind::Or,
            };
            let mut trees = bool_op.values.iter().map(|value| condition_tree_of(value, place));
            let mut folded = trees.next()??;
            for next in trees {
                folded = and_or_tree(kind, folded, next?);
            }
            Some(folded)
        }
        Expr::Compare(compare) => Some(compare_tree_of(compare, place)),
        Expr::Call(call) => Some(call_leaf_tree_of(call, place)),
        _ => Some(other_tree()),
    }
}

/// `ExprCompare` → a `NarrowTree`: a chained comparison folds to the
/// `And` of its adjacent pairs (same CPython citation the VALUES
/// channel's `narrow_compare` follows) — this reading does not depend
/// on `truth` the way the VALUES channel's falsity short-circuit does,
/// since the kernel's own answer already carries BOTH the `whenTrue`
/// (the chain held) and `whenFalse` (the chain's negation — a
/// disjunction over which pair failed, which the kernel proves
/// directly rather than this file approximating it as "narrows
/// nothing").
pub(super) fn compare_tree_of(compare: &ruff_python_ast::ExprCompare, place: &str) -> NarrowTree {
    let mut left = compare.left.as_ref();
    let mut folded: Option<NarrowTree> = None;
    for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
        let leaf = comparison_leaf_tree_of(left, *op, right, place);
        folded = Some(match folded {
            Some(existing) => and_or_tree(NarrowTreeKind::And, existing, leaf),
            None => leaf,
        });
        left = right;
    }
    folded.unwrap_or_else(other_tree)
}

/// One comparison pair (`left op right`) → a `NarrowTree` leaf, scoped
/// to `place`: a numeric literal on the other side lowers to `Cmp`
/// (`<`/`<=`/`>`/`>=`) or `Eq`/`not Eq` (`==`/`!=`); a literal
/// collection on the right of `in`/`not in` lowers to the membership
/// fold (`membership_leaf_tree_of`); anything else (`is`/`is not` —
/// read separately by the VALUES channel only, since `None` is never a
/// `Kind::Set` member; two changing names) is `other_tree()`.
pub(super) fn comparison_leaf_tree_of(left: &Expr, op: CmpOp, right: &Expr, place: &str) -> NarrowTree {
    // `place in <collection>` / `place not in <collection>` — membership
    // against a literal collection of scalars, folded to the DISJUNCTION
    // of its members' own equality leaves (see `membership_leaf_tree_of`).
    if matches!(op, CmpOp::In | CmpOp::NotIn) {
        if name_of(left) != Some(place) {
            return other_tree();
        }
        let Some(leaf) = membership_leaf_tree_of(right) else {
            return other_tree();
        };
        return if op == CmpOp::In { leaf } else { not_tree(leaf) };
    }
    // A STRING-literal equality (`layout == "horizontal"`) lowers to the
    // kernel's own EqSeq leaf — the word's code points ride `points`
    // (set_functions/narrow.lean's `.eqSeq`), so a string-tuple-union
    // set (a `Literal["…", …]` alias) narrows to the named member on
    // the when-true side and its complement on the when-false side.
    // `!=` is the same leaf under Not. Ordering (`<` etc.) over strings
    // has no kernel leaf and stays `other_tree()` below.
    if matches!(op, CmpOp::Eq | CmpOp::NotEq) {
        let word = if name_of(left) == Some(place) {
            string_literal_points(right)
        } else if name_of(right) == Some(place) {
            string_literal_points(left)
        } else {
            None
        };
        if let Some(points) = word {
            let leaf = NarrowTree { kind: NarrowTreeKind::EqSeq, points, ..other_tree() };
            return if op == CmpOp::Eq { leaf } else { not_tree(leaf) };
        }
    }
    // `math.copysign(<positive literal>, place) == <literal>` (or `!=`)
    // — pow.8's own sign-bit read (`math_models.rs::copysign_call`'s
    // doc: "IEEE 754's copysign operation... the RESULT'S sign matches
    // the sign of y", `f64::copysign`, including the signed-zero corner
    // `copysign(1.0, -0.0) == -1.0`), read here as a `Cmp` leaf on
    // `place` itself: the comparison's own truth states `place`'s sign
    // bit directly, one order weaker than the exact `copysign_call`
    // value (a WINDOW, not a point) but the only shape the kernel's
    // `Cmp`/`Eq` grammar can carry through a comparison leaf. `place`'s
    // sign bit matching the magnitude's sign means `place >= 0` (the
    // magnitude side stays a POSITIVE literal by convention across this
    // corpus, so a matching sign reads as nonnegative, including +0.0 —
    // never `> 0`, since `copysign(1.0, +0.0) == 1.0` too); the opposite
    // sign reads as `place <= 0` (including −0.0, the sign-bit mirror).
    // Checked BEFORE the bare-name arms below, which would otherwise
    // read `name_of(left)`/`name_of(right)` as `None` for a `Call` and
    // fall through to `other_tree()` — exactly the gap that left
    // `sign_positive_inside` (A2.guard.sort) unnarrowed.
    if let Some(leaf) = copysign_sign_leaf_tree_of(left, op, right, place) {
        return leaf;
    }
    let Some(numeric_op) = numeric_cmp_op(op) else {
        return other_tree();
    };
    // `<place> ± k1 <op> k2` (`n - 1 >= 0`, B1.keep.join's own ternary
    // guard) — the tested side names no bare `place`, but an AFFINE SHIFT
    // of it (`affine_place_of`'s own doc). Read BEFORE the bare-name arms
    // below (which would otherwise silently fall through to `other_tree()`
    // for this shape, exactly the gap that left the ternary's own arms
    // unnarrowed): folding the shift's own literal into the comparison's
    // literal by the inverse operation turns "n - 1 >= 0" into "n >= 1",
    // a claim `place` itself, letting this leaf build the SAME `Cmp`/`Eq`
    // tree the bare-name arms below build. Checked on EITHER side (`n - 1
    // >= 0` or `0 <= n - 1`), mirroring `mirror_cmp_op`'s own two-sided
    // reading for a bare name.
    if let Some((on_place, literal)) = affine_comparison_literal(left, right, place) {
        let effective_op = if on_place { numeric_op } else { mirror_cmp_op(numeric_op) };
        return numeric_comparison_tree(effective_op, literal);
    }
    let (on_place, literal) = if name_of(left) == Some(place) {
        (true, literal_number(right))
    } else if name_of(right) == Some(place) {
        (false, literal_number(left))
    } else {
        return other_tree();
    };
    let Some(literal) = literal else {
        return other_tree();
    };
    let effective_op = if on_place { numeric_op } else { mirror_cmp_op(numeric_op) };
    numeric_comparison_tree(effective_op, literal)
}

/// `math.copysign(<positive literal>, place) == <literal>` (or `!=`) as
/// a `Cmp` leaf on `place`'s own sign bit — `None` for any other shape:
/// the call side must be exactly `math.copysign(<positive number
/// literal>, <place>)` (the magnitude a plain literal, never `place`
/// itself — this leaf reads only the SIGN half of `copysign`, not the
/// magnitude, so a `place`-valued magnitude states nothing this leaf
/// can fold), and the other side of the comparison a plain number
/// literal. `==`/`!=` are the only operators this construct's own
/// fixture ever spells (`copysign(...) == 1.0`) and the only ones a
/// sign READ naturally composes with; any other operator (`<`, `>`,
/// …) narrows nothing here.
///
/// The comparison literal's sign against the magnitude's sign decides
/// the direction: matching signs (`copysign(1.0, place) == 1.0`) state
/// `place >= 0` (nonnegative — IEEE 754 zero has no sign for THIS
/// purpose: `copysign(1.0, +0.0)` is `+1.0`, so a positive-literal
/// magnitude admits `place == +0.0` on its matching side); opposite
/// signs state `place <= 0`, the mirror. `!=` negates the same
/// `GtE`/`LtE` claim through `numeric_comparison_tree`'s own `Cmp`
/// path by asking the OPPOSITE-sign question instead (`copysign(x,
/// place) != 1.0` is exactly `copysign(x, place) == -1.0` over the
/// two-value sign range), so both operators route through the one
/// `Cmp` builder below with no separate `Eq` leaf needed — a sign has
/// only two values, so "not this sign" already IS "the other sign."
pub(super) fn copysign_sign_leaf_tree_of(left: &Expr, op: CmpOp, right: &Expr, place: &str) -> Option<NarrowTree> {
    if !matches!(op, CmpOp::Eq | CmpOp::NotEq) {
        return None;
    }
    let (magnitude_sign, other_side) = if let Some(sign) = copysign_call_place_magnitude_sign(left, place) {
        (sign, right)
    } else if let Some(sign) = copysign_call_place_magnitude_sign(right, place) {
        (sign, left)
    } else {
        return None;
    };
    let compared_value = literal_number(other_side)?;
    let same_sign = (compared_value >= 0.0) == (magnitude_sign >= 0.0);
    let proves_nonnegative = same_sign == (op == CmpOp::Eq);
    let cmp_op = if proves_nonnegative { NumericCmpOp::GtE } else { NumericCmpOp::LtE };
    Some(cmp_tree(narrow_cmp_op_of(cmp_op).expect("GtE/LtE always map"), 0.0))
}

/// Whether `expression` is exactly `math.copysign(<positive number
/// literal>, place)` — `place`'s own name in the SIGN-SOURCE argument
/// position, paired with the magnitude literal's sign. `None` for any
/// other callee, argument count, keyword argument, or a magnitude that
/// is not a plain number literal (a `place`-valued or computed
/// magnitude states no fixed sign this leaf can read).
pub(super) fn copysign_call_place_magnitude_sign(expression: &Expr, place: &str) -> Option<f64> {
    let Expr::Call(call) = expression else { return None };
    let Expr::Attribute(attribute) = call.func.as_ref() else { return None };
    if attribute.attr.as_str() != "copysign" {
        return None;
    }
    if !matches!(attribute.value.as_ref(), Expr::Name(module) if module.id.as_str() == "math") {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [magnitude, sign_source] = call.arguments.args.as_ref() else { return None };
    if name_of(sign_source) != Some(place) {
        return None;
    }
    literal_number(magnitude)
}

/// The `NarrowTree` a numeric comparison's own effective operator and
/// literal build — `Eq`/`NotEq` fold to the kernel's own `Eq` leaf (never
/// `Cmp`, which carries only the four order operators), every other
/// operator folds to `Cmp`. Shared by `comparison_leaf_tree_of`'s bare-name
/// arm and its affine-shift arm so the two build the identical tree shape
/// once the effective operator and literal are known.
pub(super) fn numeric_comparison_tree(effective_op: NumericCmpOp, literal: f64) -> NarrowTree {
    match effective_op {
        NumericCmpOp::Eq => NarrowTree { kind: NarrowTreeKind::Eq, k: literal, ..other_tree() },
        NumericCmpOp::NotEq => not_tree(NarrowTree { kind: NarrowTreeKind::Eq, k: literal, ..other_tree() }),
        _ => {
            let kernel_op = narrow_cmp_op_of(effective_op).expect("Eq/NotEq handled above");
            cmp_tree(kernel_op, literal)
        }
    }
}

/// Whether `expression` is `<place> + k` or `<place> - k` — a literal
/// AFFINE SHIFT of the tested place (`n - 1`, `n + 1`), for a literal `k`
/// this file already reads (`literal_number`). The shift amount `k`, or
/// `None` for any other shape (a bare name, a shift of a DIFFERENT name,
/// two changing names, a non-literal offset).
pub(super) fn affine_shift_of_place(expression: &Expr, place: &str) -> Option<f64> {
    let Expr::BinOp(binop) = expression else {
        return None;
    };
    if name_of(&binop.left) != Some(place) {
        return None;
    }
    let offset = literal_number(&binop.right)?;
    match binop.op {
        ruff_python_ast::Operator::Add => Some(offset),
        ruff_python_ast::Operator::Sub => Some(-offset),
        _ => None,
    }
}

/// The BASE NAME inside an affine shift (`n - 1` names `n`), for whichever
/// name sits there — `collect_names`'s own place collector needs this
/// (unlike `affine_shift_of_place`, which is only asked once `place` is
/// already known) to discover that a comparison like `n - 1 >= 0` is
/// relevant to `n` at all, before the SET channel's per-name loop can ask
/// `condition_tree_of` to build a tree relative to it. `None` for any
/// other shape (a bare name — read separately by `name_of` — a
/// non-literal offset, an operator other than `+`/`-`).
pub(super) fn affine_shifted_name_of(expression: &Expr) -> Option<&str> {
    let Expr::BinOp(binop) = expression else {
        return None;
    };
    if !matches!(binop.op, ruff_python_ast::Operator::Add | ruff_python_ast::Operator::Sub) {
        return None;
    }
    literal_number(&binop.right)?;
    name_of(&binop.left)
}

/// One comparison pair's own AFFINE-SHIFT reading: `<place> ± k1 <op> k2`
/// (`n - 1 >= 0`) or the mirrored `k2 <op> <place> ± k1` (`0 <= n - 1`) —
/// `place` sits inside an affine shift on one side, a plain literal on the
/// other. Answers `(on_place, effective_literal)`: `on_place` tells the
/// caller which side `place`'s own shift sits on (so it can still mirror
/// the comparison operator the same way the bare-name arm does), and
/// `effective_literal` is the comparison's own literal with the shift's
/// offset folded in — "n - 1 >= 0" is exactly "n >= 0 + 1", so the shift
/// (`-1`) is subtracted back out: `effective_literal = other_literal -
/// shift`. `None` when neither side is an affine shift of `place`, or the
/// OTHER side is not a plain literal (a shift compared to a second
/// changing expression states no single-literal claim this leaf can fold).
pub(super) fn affine_comparison_literal(left: &Expr, right: &Expr, place: &str) -> Option<(bool, f64)> {
    if let Some(shift) = affine_shift_of_place(left, place) {
        let other = literal_number(right)?;
        return Some((true, other - shift));
    }
    if let Some(shift) = affine_shift_of_place(right, place) {
        let other = literal_number(left)?;
        return Some((false, other - shift));
    }
    None
}

/// A plain string literal's code points, one f64 per point — the word
/// an `EqSeq` leaf carries. Any other expression (an f-string, a
/// concatenation, a name) is `None`; only the literal's own spelling
/// is a proved word.
pub(super) fn string_literal_points(expr: &Expr) -> Option<Vec<f64>> {
    let Expr::StringLiteral(literal) = expr else {
        return None;
    };
    Some(literal.value.to_str().chars().map(|c| c as u32 as f64).collect())
}

/// `place in <collection>` as a `NarrowTree`: a literal list/tuple/set
/// of scalars folds to the DISJUNCTION of its members' own equality
/// leaves — `x in [1, 2, 3]` becomes `Or(Eq 1, Or(Eq 2, Eq 3))`.
///
/// That fold IS membership under the kernel's own `narrowQ`
/// (`set_functions/narrow.lean`), on both sides at once, with no new
/// leaf needed. Truth: `orClaim` unions the members' singletons into
/// exactly the one-of set, STRONG (every disjunct's own truth claim is
/// strong, and `orClaim` keeps strength only when all are — each `Eq`
/// leaf's holding proves the value real, so the union does too).
/// Falsity: `andClaim` intersects the members' own real-difference
/// claims, WEAK — ℝ̄ minus every listed value, which is precisely what
/// `not in` proves for a value already known real. Both claims are the
/// ones `narrowQ_sound` already proves; the fold buys the whole `in`
/// vocabulary at the existing soundness theorem's price.
///
/// The kernel's `inSet` leaf is NOT the route here: it is a SEQUENCE-
/// world leaf (`leafEval`'s own `.inSet _, _, _ => False` gives it no
/// scalar runs, and its falsity claim is `diffSet stringsSet S` — C*
/// minus the set, a claim about strings). A numeric place tested for
/// membership belongs in the scalar world, and the `Eq`/`Or` fold puts
/// it there.
///
/// Members must share ONE sort — all numeric, or all string — matching
/// the boundary's own refusal to mix the worlds in one tree
/// (`exports_narrow.lean`'s `treeScalarClaim && treeSeqClaim` check,
/// which FAILS the whole question for a mixed tree). A mixed or empty
/// collection, a non-literal collection (a name, a comprehension, a
/// call), or any member this file cannot read as a literal, answers
/// `None` — the caller lowers `other_tree()`, narrowing nothing.
///
/// A DICT is never read: `x in {...}` tests the dict's KEYS, and a
/// `dict` display's keys are a different collection from the members a
/// list display names. Declining is conservative, never wrong.
pub(super) fn membership_leaf_tree_of(collection: &Expr) -> Option<NarrowTree> {
    let elements: &[Expr] = match collection {
        Expr::List(list) => &list.elts,
        Expr::Tuple(tuple) => &tuple.elts,
        Expr::Set(set) => &set.elts,
        _ => return None,
    };
    if elements.is_empty() {
        return None;
    }
    // one sort per collection: read every member as a number, or every
    // member as a string, and decline the moment the two mix
    let leaves: Option<Vec<NarrowTree>> = elements
        .iter()
        .map(|element| {
            literal_number(element)
                .map(|k| NarrowTree { kind: NarrowTreeKind::Eq, k, ..other_tree() })
                .or_else(|| {
                    string_literal_points(element)
                        .map(|points| NarrowTree { kind: NarrowTreeKind::EqSeq, points, ..other_tree() })
                })
        })
        .collect();
    let leaves = leaves?;
    let all_numeric = leaves.iter().all(|leaf| leaf.kind == NarrowTreeKind::Eq);
    let all_words = leaves.iter().all(|leaf| leaf.kind == NarrowTreeKind::EqSeq);
    if !all_numeric && !all_words {
        return None;
    }
    let mut folded = None;
    for leaf in leaves {
        folded = Some(match folded {
            Some(existing) => and_or_tree(NarrowTreeKind::Or, existing, leaf),
            None => leaf,
        });
    }
    folded
}

/// The SET channel's own dispatcher for a bare `Expr::Call` test:
/// `<place>.is_integer()` reads as the `IsInt AND IsFinite` leaf
/// (`is_integer_leaf_tree_of`'s own doc); every other call, INCLUDING
/// `isinstance(...)`, is `other_tree()` (`isinstance_leaf_tree_of`'s own
/// doc — a sort claim the kernel's COMPARISON/MEMBERSHIP vocabulary
/// cannot further express about a Set already scoped to that sort).
pub(super) fn call_leaf_tree_of(call: &ruff_python_ast::ExprCall, place: &str) -> NarrowTree {
    if let Some(leaf) = is_integer_leaf_tree_of(call, place) {
        return leaf;
    }
    isinstance_leaf_tree_of(call, place)
}

/// `<place>.is_integer()` as a `NarrowTree` leaf, `None` for any other
/// call shape (a different method name, a non-empty argument list, or a
/// receiver that is not the bare name `place`) — the SET-channel twin of
/// `expressions.rs`'s own single-known-value `is_integer` row
/// (`stdtypes.rst`, `float.is_integer()`: "Return True if the float
/// instance is finite with integral value, and False otherwise"). A
/// `place` currently bound `Kind::Set` has no one value to test that way,
/// so this states the SAME two-part claim as a kernel leaf instead:
/// `IsInt` (integral within ℝ̄) AND `IsFinite` (excludes both
/// infinities) — `is_integer()` on `float('inf')` is `False` precisely
/// because the finite half fails, so `IsInt` alone would overclaim on an
/// unbounded-above parameter like `to_page_size`'s own `x: float`
/// (showcase.py) the way this leaf's own construct citation names.
pub(super) fn is_integer_leaf_tree_of(call: &ruff_python_ast::ExprCall, place: &str) -> Option<NarrowTree> {
    let Expr::Attribute(attribute) = call.func.as_ref() else { return None };
    if attribute.attr.as_str() != "is_integer" {
        return None;
    }
    if !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return None;
    }
    if name_of(&attribute.value) != Some(place) {
        return None;
    }
    let is_int = NarrowTree { kind: NarrowTreeKind::IsInt, ..other_tree() };
    let is_finite = NarrowTree { kind: NarrowTreeKind::IsFinite, ..other_tree() };
    Some(and_or_tree(NarrowTreeKind::And, is_int, is_finite))
}

/// `isinstance(place, ...)` as a `NarrowTree` leaf: the kernel's own
/// grammar has no "Python sort" leaf (`IsInt` tests integrality within
/// ℝ̄, not "is this Python `int`") — a sort claim is entirely this
/// file's own `narrow_isinstance_call`/`sort_seed` job, on the VALUES
/// channel or at seeding time, never the kernel's. `other_tree()`
/// always: an `isinstance` test says nothing further the kernel's
/// COMPARISON/MEMBERSHIP vocabulary can express about a Set already
/// scoped to that sort.
pub(super) fn isinstance_leaf_tree_of(_call: &ruff_python_ast::ExprCall, _place: &str) -> NarrowTree {
    other_tree()
}

/// Whether `tree` states anything at all — an all-`Other` tree asks
/// the kernel a question with no answer worth having (`refined-ts-go`'s
/// own `SaysAnything`).
pub(super) fn says_anything(tree: &NarrowTree) -> bool {
    match tree.kind {
        NarrowTreeKind::Other => false,
        NarrowTreeKind::Not => says_anything(tree.a.as_deref().expect("not carries A")),
        NarrowTreeKind::And | NarrowTreeKind::Or => {
            says_anything(tree.a.as_deref().expect("and/or carries A"))
                || says_anything(tree.b.as_deref().expect("and/or carries B"))
        }
        _ => true,
    }
}

/// Meets a kernel narrowing claim into `current`'s own set: the
/// INTERSECTION of `current.set`'s forms with `claim_set`'s forms
/// (`RefinedSet`'s forms conjoin — the same reading
/// `refined_domain::lattice_operations::meet_known`'s own Set×Set
/// branch takes), keeping `current`'s `kind_tag` (the kernel's claim
/// carries no Python sort tag of its own) and `current`'s own trust
/// grade (never claimed stronger by a narrowing than the value that
/// flowed in — `loops.rs`'s `kernel_bounded_counter_environment` binds
/// its own kernel answer at the SAME grade the entry binding carried,
/// the matching precedent).
pub(super) fn meet_set_answer(current: &AbstractValue, claim_set: &RefinedSet) -> AbstractValue {
    let mut combined = current.set.forms.clone();
    combined.extend(claim_set.forms.clone());
    let grade = trust_level_of(current);
    AbstractValue {
        kind_tag: current.kind_tag,
        ..known_set(make_refined_set(combined), None, grade, current.set_kind_tag)
    }
}
