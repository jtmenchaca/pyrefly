//! The cross-adapter conformance battery: the mechanical catcher for
//! the defect class where one adapter's compiled set diverges from
//! another's for the same declaration. The measured instance this
//! battery exists to catch: a pattern-only string alias compiled with
//! a redundant whole-string ground on one side and clean on the
//! other, blinding the kernel's prover on only that side
//! (`surface.rs`'s own `pattern_only_alias_drops_the_redundant_
//! string_ground` test; `chain_method.go`'s own "regex" case comment
//! on the TS side).
//!
//! This file is the CONSUMER half of the battery: the Go twin
//! (`packages/refinedts/refined-ts-go/internal/refinedts/conformance/
//! cross_adapter_twins_test.go`) compiles each row's TS z-surface
//! spelling through the real annotations machinery and WRITES its
//! wire JSON to a shared golden file; this file compiles the SAME
//! row's Python `Annotated`/`Field` spelling through `surface.rs`'s
//! real `compile_aliases`, encodes it through the identical wire
//! codec (`refined_kernel::wire_format::wire_set`), and asserts BYTE-
//! IDENTICAL JSON against that golden.
//!
//! The table (`python_twin_table`) carries one row per shape the two
//! adapters must read identically — a pattern-only alias, the
//! timestamp grammar, a string length window, a numeric literal
//! union, a string literal union, a numeric window with
//! int/multipleOf, and an Optional (None-admitting) scalar — each
//! row's `golden_name` naming the file `<name>.json` under the shared
//! testdata directory both languages reach.
//!
//! A MISMATCH HERE MEANS ONE ADAPTER'S COMPILE DRIFTED. Fix the
//! compile, never the golden — unless the kernel's own grammar
//! changed, in which case both sides' compiles and the Go twin's
//! golden all move together, in the same batch. This file never
//! writes the golden; only the Go twin does (see that file's own
//! doc for why Go is upstream here).
//!
//! The battery's first runs caught, and the compiles then fixed, two
//! live divergences (the mechanical catch this file exists to
//! perform): the TS numeric chain carried a redundant unbounded ray
//! beside its tighter bound (fixed — chain_numeric_method.go now runs
//! CanonicalScalarForms exactly as the string chain runs
//! WithoutStringGround), and the TS union of numeric literals compiled
//! nested where Python's Literal[...] compiles one flat oneOf (fixed —
//! chain_root_constructor.go's sort-gated MergeScalarOneOfArms). The
//! remaining spelling rule both sides follow: canonical scalar form
//! ORDER is rays first, then integer, then multipleOf (surface.rs's
//! declaration-compile sort; the Go golden's own order).

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use refined_kernel::wire_format::wire_set;
    use refined_sets::refinement_forms::RefinedSet;
    use ruff_python_ast::ModModule;
    use serde_json::Value;

    use crate::surface::compile_aliases;

    /// The shared golden-file directory both this crate and the Go
    /// twin reach by a short relative path — a new top-level
    /// directory under `packages/` rather than inside either
    /// adapter's own tree, since the fixture belongs to neither
    /// adapter alone. From this file's own directory
    /// (`packages/refinedpy/pyrefly/crates/refinedpy/src/`), five
    /// levels up reaches `packages/`.
    fn twins_testdata_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../conformance-twins/testdata")
    }

    fn parsed(source: &str) -> ModModule {
        ruff_python_parser::parse_module(source)
            .expect("fixture source parses")
            .into_syntax()
    }

    /// One declaration twin's Python half: the module source carrying
    /// exactly one top-level alias named `X`, and the golden file
    /// name (matching the Go twin's `twinRow.name` for the same row)
    /// this alias's compiled set must match byte-for-byte.
    struct PythonTwinRow {
        golden_name: &'static str,
        module_source: &'static str,
        /// true when this row's own alias states `admits_none` — the
        /// out-of-band bit the golden's own `"absent"` field carries,
        /// compared separately from the wire set exactly as the Go
        /// twin's `Annotation.Absent` does.
        admits_none: bool,
        about: &'static str,
    }

    /// The Python twin of every row in the Go twin's `twinTable()` —
    /// same order, same `golden_name`s. See each row's own `about`
    /// for which of the brief's named divergent shapes it exercises.
    fn python_twin_table() -> Vec<PythonTwinRow> {
        vec![
            PythonTwinRow {
                golden_name: "pattern-only-alias",
                module_source: "from pydantic import Field\n\
                     from typing import Annotated\n\
                     type X = Annotated[str, Field(pattern=r\"^[0-9]+$\")]\n",
                admits_none: false,
                about: "a pattern-only string alias: the regex conjunct alone, no redundant C* ground riding beside it",
            },
            PythonTwinRow {
                golden_name: "timestamp-grammar",
                module_source: "from pydantic import Field\n\
                     from typing import Annotated\n\
                     type X = Annotated[str, Field(pattern=r\"(?:(?:\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12]\\d|3[01])|(?:0[469]|11)-(?:0[1-9]|[12]\\d|30)|(?:02)-(?:0[1-9]|1\\d|2[0-8])))T(?:(?:[01]\\d|2[0-3]):[0-5]\\d(?::[0-5]\\d(?:\\.\\d+)?)?(?:Z))$\")]\n",
                admits_none: false,
                about: "the timestamp grammar: the identical ISO-datetime regex text the TS twin spells as .regex(<same text>), on a pydantic pattern= alias",
            },
            PythonTwinRow {
                golden_name: "string-length-window",
                module_source: "from pydantic import Field\n\
                     from typing import Annotated\n\
                     type X = Annotated[str, Field(min_length=1, max_length=10)]\n",
                admits_none: false,
                about: "a min/max length window: one bounded repetition form over the codepoint ground",
            },
            PythonTwinRow {
                golden_name: "numeric-literal-union",
                module_source: "from typing import Literal\n\
                     type X = Literal[1, 2, 3]\n",
                admits_none: false,
                about: "a numeric literal union: one oneOf([1,2,3]) form",
            },
            PythonTwinRow {
                golden_name: "string-literal-union",
                module_source: "from typing import Literal\n\
                     type X = Literal[\"a\", \"b\"]\n",
                admits_none: false,
                about: "a string literal union: the union of each member's own singleton tuple",
            },
            PythonTwinRow {
                golden_name: "numeric-window-int-multiple-of",
                module_source: "from pydantic import Field\n\
                     from typing import Annotated\n\
                     type X = Annotated[int, Field(ge=0, le=100, multiple_of=5)]\n",
                admits_none: false,
                about: "a numeric window with an integer floor and a multipleOf step: integer + atLeast(0) + atMost(100) + multipleOf(5)",
            },
            PythonTwinRow {
                golden_name: "optional-scalar-base-set",
                module_source: "from pydantic import Field\n\
                     from typing import Annotated, Optional\n\
                     type X = Optional[Annotated[int, Field(ge=0)]]\n",
                admits_none: true,
                about: "an Optional (None-admitting) scalar: the INNER base set only (atLeast(0)) — admits_none rides out of band, compared separately from the wire set",
            },
        ]
    }

    /// Reads `<golden_name>.json`'s `"set"` field back into a
    /// `serde_json::Value` — the same envelope shape the Go twin's
    /// `twinGoldenJSON` writes (`{"set": <wire JSON>, "absent": bool}`),
    /// and the `"absent"` field alongside it.
    fn read_golden(golden_name: &str) -> (Value, bool) {
        let path = twins_testdata_dir().join(format!("{golden_name}.json"));
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!(
                "reading golden {path:?}: {err} — run the Go twin's \
                 TestCrossAdapterTwinsWriteTheGoldenWireJSON first"
            )
        });
        let parsed: Value = serde_json::from_str(&raw)
            .unwrap_or_else(|err| panic!("parsing golden {path:?} as JSON: {err}"));
        let set = parsed
            .get("set")
            .unwrap_or_else(|| panic!("golden {path:?} carries no \"set\" field"))
            .clone();
        let absent = parsed
            .get("absent")
            .and_then(Value::as_bool)
            .unwrap_or_else(|| panic!("golden {path:?} carries no \"absent\" field"));
        (set, absent)
    }

    /// One Python twin row's own compiled set, read through
    /// `compile_aliases` exactly as the checker itself reads a
    /// module's top-level aliases — the alias must be named `X`,
    /// matching every row's `module_source` above.
    fn compiled_python_set(row: &PythonTwinRow) -> (RefinedSet, bool) {
        let module = parsed(row.module_source);
        let aliases = compile_aliases(&module);
        let entry = aliases
            .get("X")
            .unwrap_or_else(|| panic!("{}: alias X did not compile", row.golden_name));
        (entry.set.clone(), entry.admits_none)
    }

    /// The battery: every row's Python spelling compiles to the
    /// EXACT wire JSON the Go twin already wrote for the same row,
    /// and the same admits-none bit.
    #[test]
    fn test_every_python_twin_matches_the_go_adapters_golden_wire_json() {
        let mut compared: HashMap<&str, ()> = HashMap::new();
        for row in python_twin_table() {
            let (golden_set, golden_absent) = read_golden(row.golden_name);
            let (compiled_set, compiled_admits_none) = compiled_python_set(&row);
            let compiled_wire = wire_set(&compiled_set);

            assert_eq!(
                row.admits_none, golden_absent,
                "{}: the row declares admits_none = {} but the golden's own \"absent\" field says \
                 {golden_absent} — the golden on disk is not the row this table describes",
                row.golden_name, row.admits_none
            );

            assert_eq!(
                compiled_wire, golden_set,
                "{}: the Python compile's wire JSON diverged from the Go adapter's golden ({}). \
                 A mismatch here means one adapter's compile drifted — fix the compile, never \
                 the golden, unless the kernel's own grammar changed.",
                row.golden_name, row.about
            );
            assert_eq!(
                compiled_admits_none, golden_absent,
                "{}: admits_none = {compiled_admits_none}, want {golden_absent} (the Go twin's \
                 Annotation.Absent) — the two adapters disagree on whether this declaration \
                 admits the absent value ({})",
                row.golden_name, row.about
            );
            compared.insert(row.golden_name, ());
        }
        assert_eq!(compared.len(), python_twin_table().len());
    }

    /// `loaded_kernel` mirrors `lattice_conformance.rs`'s own test
    /// helper: a missing dylib artifact prints to stderr and the
    /// caller returns early, never failing the run.
    fn loaded_kernel() -> Option<std::sync::Arc<refined_kernel::kernel_interface::RefinedTSKernel>> {
        let path = refined_kernel::kernel_bridge::dylib_path();
        if !refined_kernel::kernel_bridge::kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(refined_kernel::kernel_bridge::load_kernel(&path).expect("load_kernel"))
    }

    /// The SECOND half of the brief's design: "plus the kernel
    /// question both must answer identically." A self-containment
    /// sanity check (every Python twin's own compiled set must
    /// contain itself) — a scaffold ready for the true cross-language
    /// version once the Go twin's own compiled set crosses the wire
    /// for a live `scalar_subset` ask against this side's set,
    /// matching the Go twin's own
    /// `TestCrossAdapterTwinsAgreeWithTheKernelOnMutualContainment`.
    #[test]
    fn test_every_python_twin_agrees_with_the_kernel_on_self_containment() {
        let Some(kernel) = loaded_kernel() else { return };
        for row in python_twin_table() {
            let (compiled_set, _) = compiled_python_set(&row);
            let contains_itself =
                crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&compiled_set, &compiled_set));
            assert!(
                matches!(contains_itself, Ok(true)),
                "{}: kernel.scalar_subset(set, set) = {contains_itself:?}, want Ok(true) — a set must \
                 always contain itself",
                row.golden_name
            );
        }
    }
}
