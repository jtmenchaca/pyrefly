//! Sequence-shape recognition and the kernel reread-safety gate.

use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;

use super::scalar::within_codepoint_door;

/// A `Star`/`Repeat`/`RepeatWord` form's own element sits inside the
/// codepoint alphabet — this crate's grammar reuses `Star`/`Repeat` for a
/// NUMERIC element too (`check.rs::seed_parameters`'s `list[int]`/
/// `set[int]`/`Sequence[int]` parameter seed, `Form::Star(int's own
/// set)`), so a bare repetition form is string-shaped only when its
/// element demonstrably IS codepoints, never merely because it wears one
/// of these forms.
///
/// Two spellings pass: `is_character` — the element IS the whole
/// alphabet exactly (a plain `list[str]`/`Sequence[str]` element seed) —
/// or `within_codepoint_door` — the element is a NARROWER codepoint-only
/// subset, the shape a `.regex(...)`-compiled character class actually
/// produces (`regex_compiler.rs`'s `code_range`/`character_class`:
/// `[a-z]` compiles to `Integer ∧ AtLeast(0x61) ∧ AtMost(0x7A)`, an
/// `Integer`-window wholly inside the codepoint alphabet, never the full
/// alphabet itself). Before this second spelling, a `Repeat`/`Star` over
/// a narrowed class (`LabelPattern = Annotated[str,
/// Field(pattern=r"^[a-z]{3,8}$")]`'s own compiled `Repeat`) read as
/// NEITHER string- nor number-shaped by this file's own two tests, so
/// `judge`'s `sequence_question` gate never tried `seq_subset` at all and
/// fell straight to `scalar_subset`, which the kernel refuses outright
/// for a non-1-tuple shape — the undetermined `g-strings-and-formats.py`
/// row this fixes (`kernel_declined_containment`, RTS7002).
pub(super) fn repetition_element_is_codepoints(form: &refined_sets::refinement_forms::Refinement) -> bool {
    let one = refined_sets::refinement_forms::make_refined_set(vec![form.clone()]);
    match refined_sets::repetition_window_forms::as_repetition(&one) {
        Some(repeated) => {
            refined_sets::format_string_shapes::is_character(&repeated.element)
                || within_codepoint_door(&repeated.element, false)
        }
        None => false,
    }
}

pub(crate) fn sequence_shaped(set: &refined_sets::refinement_forms::RefinedSet) -> bool {
    use refined_sets::refinement_forms::Form;
    !set.forms.is_empty()
        && set.forms.iter().all(|form| match form.form {
            // EmptyTuple/Concatenation/Word carry no separate "element
            // sort" of their own (an EmptyTuple names no element at all; a
            // Concatenation's operands are themselves nested sets this
            // crate only ever builds over codepoints — `string_tuple`'s
            // own encoding; a Word's codepoints are a literal by
            // construction) — string-shaped unconditionally.
            Form::EmptyTuple | Form::Concatenation | Form::Word => true,
            Form::Star | Form::Repeat | Form::RepeatWord => repetition_element_is_codepoints(form),
            Form::Union | Form::Difference => {
                form.a_.as_deref().map(sequence_shaped).unwrap_or(false)
                    && form.b.as_deref().map(sequence_shaped).unwrap_or(false)
            }
            Form::AtLeast
            | Form::Above
            | Form::AtMost
            | Form::Below
            | Form::Integer
            | Form::MultipleOf
            | Form::OneOf => false,
        })
}

/// `sequence_shaped`'s own reread-safety gate, asked of the kernel FIRST
/// — the determination parity `seq_no_scalar_reread`
/// (`refined_seq_no_scalar_reread`, proved by `noScalarRereadF_sound`)
/// exists for: the kernel's own recursion requires BOTH of a
/// `Concatenation`'s operands to themselves recurse-prove reread-safe
/// (`noScalarRereadFormF`'s `.Concatenation A B => noScalarRereadF A &&
/// noScalarRereadF B`, `set_functions/no_scalar_reread.lean`), while
/// `sequence_shaped`'s own `Form::Concatenation => true` arm above admits
/// ANY concatenation outright, without inspecting operands — the same
/// gap refined-ts-go's `noScalarReread`
/// (`abstractdomain/lattice_operations.go`) already asks the kernel
/// ahead of its own `statesOnlyLongSequences` fallback to close: a
/// kernel `true` is a proved theorem for a shape the LOCAL recursion may
/// have reached through a path the kernel's own recursion also walks (a
/// union of concatenations, the kernel's everyday shape per that file's
/// own doc), trusted outright here. The kernel's own `false` is a
/// DECLINE, never a proof of unsafety (`seq_no_scalar_reread`'s own doc:
/// "a decline that proves nothing, and the caller keeps its own
/// conservative answer there") — including for a genuine unsafe 1-tuple
/// concatenation the kernel recursed into and rejected, which is why a
/// kernel `false`/refusal falls through to `sequence_shaped` unchanged
/// rather than being read as "unsafe," exactly the same non-strengthening
/// fallback refined-ts-go's own `noScalarReread` takes on its own `ok:
/// false` branch. The two judgment sites in this file that classify an
/// untagged Set's own sort (the `is_string_sorted_set` law and the
/// `sequence_question` routing gate) ask the same kernel question
/// refined-ts-go already asks there, rather than trusting the local
/// recursion alone.
pub(super) fn sequence_shaped_safely(set: &refined_sets::refinement_forms::RefinedSet, kernel: &Arc<RefinedTSKernel>) -> bool {
    if let Ok(true) = crate::kernel_ask::ask_kernel(|| (kernel.seq_no_scalar_reread)(set)) {
        return true;
    }
    sequence_shaped(set)
}

/// Whether a set's OWN top-level forms DEMONSTRABLY state a sequence —
/// a `Star`/`Concatenation`/`Repeat`/`RepeatWord`/`EmptyTuple` form
/// sits among them. Ported from refined-ts-go's `StatesSequence`
/// (walk/sequence_measures.go): a POSITIVE, non-recursive test — unlike
/// `sequence_shaped` above (which requires EVERY form, recursing
/// through Union/Difference, and serves the Object/List/Null
/// structural-mismatch law), `states_sequence` only asks whether the
/// set's own top layer carries a sequence form at all, and is what
/// gates the string-vs-numeric-ground SORT laws below: `on_one_tuple_
/// layer` alone cannot tell a numeric one-of from a union of
/// single-character string tuples (the tuple pun — `string_tuple`'s
/// length-1 encoding is bare `OneOf`, no `Concatenation` wrapper), so
/// the sort law must see an actual sequence form before it may read
/// "on the one-tuple layer" as "numeric."
pub(crate) fn states_sequence(set: &refined_sets::refinement_forms::RefinedSet) -> bool {
    use refined_sets::refinement_forms::Form;
    set.forms.iter().any(|form| {
        matches!(
            form.form,
            Form::Star | Form::Concatenation | Form::Repeat | Form::RepeatWord | Form::EmptyTuple
        )
    })
}
