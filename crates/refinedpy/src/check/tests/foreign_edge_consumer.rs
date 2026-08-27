use super::*;
use ruff_text_size::Ranged;

/* ── the per-edge foreign-override map ───────────────────────────
 *
 * `walk_body_with_self_binding`'s pending foreign-edge override is
 * keyed by the position of the STATEMENT holding the recognized
 * edge's own `json.loads(...)` consumer (`foreign_edge_overrides:
 * HashMap<usize, Vec<(TextRange, AbstractValue)>>` — a `Vec` rather
 * than a single pair per position, since a discharged crossing whose
 * return is number-sorted publishes a SECOND override at the same
 * position: the intermediate captured-stdout reading's own
 * serialized-string fact, alongside the return fact), never a single
 * unkeyed slot — two recognized crossing calls in the same body,
 * each consumed at a DIFFERENT later statement, must each publish
 * their own fact at their own consumer, with neither clobbering the
 * other's still-pending entry.
 *
 * `findings_for_module_at`'s own route to a recognized `Override`
 * goes through `foreign_edge::foreign_edge_at`, which — in every
 * `cargo test` build of this crate — routes ITS artifact read
 * through `foreign_edge.rs`'s own `#[cfg(test)] read_foreign_ts_
 * artifact`, a crate-wide swap (not scoped to that file's own test
 * module) that answers only from a `FIXTURE_ARTIFACTS` thread-local
 * private to `foreign_edge::tests` (`register_fixture_artifact` is
 * a bare `fn`, `pub(super)` at most — never reachable from this
 * module). So a real on-disk artifact, however written, can never
 * reach `Override` through `foreign_edge_at` from a test in THIS
 * file: the read always fails and the outcome is always `Decline`,
 * which never touches `foreign_edge_overrides` at all. There is no
 * seam left in `check.rs` alone to drive `Override` end to end —
 * fixing that needs either a `pub(crate)` bridge in `foreign_edge
 * .rs` (owned by the other lane) or the corpus's own end-to-end
 * fixture path once its earlier gate (missing chain artifacts, per
 * the brief) is cleared.
 *
 * `foreign_edge_consumer_position` is the one place the ORIGINAL
 * defect actually lived (the position math that used to feed a
 * single `Option` slot) — it takes no artifact, no kernel, and no
 * `Environment`, so it is exactly what these tests can pin
 * end-to-end: TWO recognized calls at different call positions
 * resolve to their OWN independent consumer positions (never one
 * clobbering the other's key), a single recognized call's consumer
 * resolves unchanged, and a call whose result is never parsed
 * anywhere later answers `None` — the same "no position to key an
 * entry under" a single unconsumed slot used to leave as `None`.
 */

/// A tiny multi-statement body — `first = <call>`; `second =
/// <call>`; `x = <parse-shaped read>`; `y = <parse-shaped read>` —
/// built from real parsed statements, so `TextRange`s are genuine
/// ranges from `ruff_python_parser`, not hand-picked offsets.
/// Returns the parsed body and, by index, `first`'s call statement,
/// `second`'s call statement, `x`'s own assignment (whose RHS range
/// stands in for a `json.loads(first.stdout)` node), and `y`'s own
/// assignment (standing in for `json.loads(second.stdout)`).
fn diamond_shaped_body() -> Vec<Stmt> {
    parsed(concat!(
        "def f():\n",
        "    first = call_a()\n",
        "    second = call_b()\n",
        "    x = first\n",
        "    y = second\n",
    ))
    .body
    .into_iter()
    .find_map(|stmt| match stmt {
        Stmt::FunctionDef(def) => Some(def.body.to_vec()),
        _ => None,
    })
    .expect("the fixture's own def body")
}

/// The RHS range of the `Assign` at `position` — the stand-in for a
/// recognized edge's own `json.loads(...)` parse-node range: what
/// matters for `foreign_edge_consumer_position` is that the range is
/// CONTAINED by exactly one later statement, not that it is
/// literally a `json.loads` call.
fn assign_value_range(body: &[Stmt], position: usize) -> TextRange {
    let Stmt::Assign(assign) = &body[position] else {
        panic!("fixture statement {position} is not an Assign");
    };
    assign.value.range()
}

/// THE DIAMOND PIN: two recognized calls (`first` at position 0,
/// `second` at position 1), each consumed by its OWN later
/// statement (`x` at position 2 reads `first`'s range, `y` at
/// position 3 reads `second`'s range) — resolving `second`'s
/// consumer position must not disturb `first`'s already-resolved
/// one, and both must resolve to their own distinct, correct
/// position. A single unkeyed slot has no analogue for "resolve
/// twice, keep both" at all; this is exactly the map's own
/// contract.
#[test]
fn two_recognized_calls_resolve_to_their_own_independent_consumer_positions() {
    let body = diamond_shaped_body();
    let first_parse_range = assign_value_range(&body, 2); // x = first
    let second_parse_range = assign_value_range(&body, 3); // y = second

    let first_consumer = foreign_edge_consumer_position(&body, 0, first_parse_range);
    let second_consumer = foreign_edge_consumer_position(&body, 1, second_parse_range);

    assert_eq!(first_consumer, Some(2), "first's own consumer must resolve to x's position");
    assert_eq!(second_consumer, Some(3), "second's own consumer must resolve to y's position, unaffected by first's resolution");

    // Keyed exactly as `walk_body_with_self_binding` keys them: two
    // independent map entries, neither overwriting the other.
    let mut overrides: HashMap<usize, TextRange> = HashMap::new();
    overrides.insert(first_consumer.unwrap(), first_parse_range);
    overrides.insert(second_consumer.unwrap(), second_parse_range);
    assert_eq!(overrides.len(), 2, "both entries must coexist in the map, keyed under their own positions");
    assert_eq!(overrides.get(&2), Some(&first_parse_range), "position 2 must still carry first's own range");
    assert_eq!(overrides.get(&3), Some(&second_parse_range), "position 3 must still carry second's own range");
}

/// THE SINGLE-EDGE REGRESSION PIN: one recognized call, consumed by
/// the very next statement — the shape the original single slot
/// already served correctly — resolves to that next position
/// unchanged.
#[test]
fn a_single_recognized_calls_consumer_resolves_to_the_very_next_statement() {
    let body = diamond_shaped_body();
    let first_parse_range = assign_value_range(&body, 2); // x = first
    let consumer = foreign_edge_consumer_position(&body, 0, first_parse_range);
    assert_eq!(consumer, Some(2), "the single recognized call's consumer must still resolve to its own position");
}

/// THE EXPIRY PIN: a recognized call whose own range is never
/// contained by any LATER statement (no consumer anywhere in the
/// body reads it) answers `None` — mirroring the single slot's own
/// behavior for an unconsumed override: nothing is inserted, so
/// nothing can later leak onto an unrelated statement.
#[test]
fn a_call_whose_range_no_later_statement_contains_has_no_consumer_position() {
    let body = diamond_shaped_body();
    // `second`'s own call range (position 1) is never READ by any
    // statement after it in this fixture (only `y`'s RHS, which is
    // the bare name `second`, contains it in the diamond fixture
    // above — here we ask about `first`'s CALL range itself, which
    // no later statement's range contains, since `x = first` reads
    // only the NAME `first`, not the call expression `call_a()`).
    let Stmt::Assign(first_assign) = &body[0] else {
        panic!("fixture statement 0 is not an Assign");
    };
    let call_range = first_assign.value.range();
    let consumer = foreign_edge_consumer_position(&body, 0, call_range);
    assert_eq!(
        consumer, None,
        "a range no later statement contains must answer no consumer position, the same way an override \
         with nothing to key never gets applied anywhere"
    );
}

/* ── the export walk's own entry_directory ───────────────────────
 *
 * `derived_return_values_at` threads `entry_directory` into the
 * SAME `WalkContext` field `findings_for_module_at` already
 * populates, so `serve_foreign_edge_at`'s `foreign_edge_at` call
 * (line ~588) resolves a relative argv target the identical way
 * during export as during an ordinary check. `register_fixture_
 * artifact` is private to `foreign_edge::tests` (unreachable from
 * here — the brief's own boundary), so these tests cannot drive a
 * recognized edge all the way to `Override`. What IS reachable: the
 * recognized-but-undischarged edge's own DECLINE sentence, which
 * `discharge_edge_premises` builds from `edge.target_path` AFTER
 * joining it against `entry_directory` when one is given
 * (foreign_edge.rs:395-403). A relative target with no directory
 * stays the bare literal the source wrote; the identical body walked
 * WITH a directory reports the JOINED (absolute) path instead — an
 * observable difference that exists only if the walk actually
 * carried `entry_directory` through to `foreign_edge_at`, which is
 * the one fact these tests pin.
 */

/// A one-def module whose only statement is a recognized (relative-
/// target) foreign edge that consumes nothing further — the walk
/// records it as `blockers`, never `values` (no `return` anywhere in
/// the body), so the blocker sentence is exactly what
/// `discharge_edge_premises` names for this decline.
fn foreign_edge_only_module() -> ModModule {
    parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "\n",
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    parsed = json.loads(result.stdout)\n",
    ))
}

/// `derived_return_values` (no directory, the pre-existing entry
/// point) leaves a relative argv target UNJOINED — the blocker names
/// the bare literal the source wrote, never an absolute path.
#[test]
fn derived_return_values_with_no_directory_names_the_bare_relative_target() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = foreign_edge_only_module();
    let derived = derived_return_values(&module, no_imports_resolver(), &kernel);
    let blocker = derived.blockers.get("f").unwrap_or_else(|| {
        panic!("expected a recorded blocker for 'f'; values = {:?}, blockers = {:?}", derived.values, derived.blockers)
    });
    assert!(
        blocker.contains("./audio_level.ts"),
        "with no entry_directory the target must stay the bare relative literal: {blocker:?}"
    );
}

/// `derived_return_values_at` WITH a directory joins the SAME
/// relative target against it before the artifact read — the
/// blocker now names the absolute (joined) path, proving the export
/// walk carried `entry_directory` into `foreign_edge_at` exactly as
/// `findings_for_module_at` already does for an ordinary check.
#[test]
fn derived_return_values_at_with_a_directory_joins_the_relative_target_before_declining() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = foreign_edge_only_module();
    let directory = std::path::Path::new("/tmp/refinedpy-export-directory-fixture");
    let derived = derived_return_values_at(&module, no_imports_resolver(), &kernel, Some(directory));
    let blocker = derived.blockers.get("f").unwrap_or_else(|| {
        panic!("expected a recorded blocker for 'f'; values = {:?}, blockers = {:?}", derived.values, derived.blockers)
    });
    // `Path::join` keeps the source's own leading "./" verbatim
    // (foreign_edge.rs's own join: `directory.join(target)` where
    // `target` is `Path::new("./audio_level.ts")`), so the joined
    // spelling is the directory plus that exact relative text, not
    // a normalized form.
    let joined = directory.join("./audio_level.ts");
    assert!(
        blocker.contains(&joined.to_string_lossy().into_owned()),
        "with entry_directory given, the target must be joined against it before the read: {blocker:?}"
    );
}

/// A module with no foreign edge at all keeps deriving its return
/// exactly the same whether or not a directory is given — threading
/// `entry_directory` through must never change a def's OWN
/// derivation when nothing in its body ever reads it.
#[test]
fn derived_return_values_at_with_a_directory_does_not_disturb_an_ordinary_def() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Age = Annotated[int, Field(ge=0)]\n",
        "\n",
        "def f(x: Age) -> Age:\n",
        "    return x\n",
    );
    let module = parsed(source);
    let no_directory = derived_return_values(&module, no_imports_resolver(), &kernel);
    let with_directory = derived_return_values_at(
        &module,
        no_imports_resolver(),
        &kernel,
        Some(std::path::Path::new("/tmp/refinedpy-export-directory-fixture")),
    );
    assert_eq!(
        format!("{:?}", no_directory.values.get("f")),
        format!("{:?}", with_directory.values.get("f")),
        "an ordinary def's derived return must be identical regardless of entry_directory"
    );
}
