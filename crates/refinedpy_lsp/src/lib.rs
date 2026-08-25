//! RefinedPy: refinement diagnostics layered onto pyrefly's own.
//!
//! RefinedPy judges values against refinement sets (which values are
//! allowed, not just what shape they have) by asking a proved Lean
//! kernel loaded from a native dylib. Its diagnostics carry
//! `source: "refinedpy"` and codes RTS7001-RTS7005, and are appended
//! after pyrefly's own diagnostics on the same read-only transaction
//! the host already validated with `Require::Everything` — the check
//! never calls `set_memory` or `run` itself.
//!
//! This crate holds no direct call sites into pyrefly's server: it
//! implements the four `RefinementHooks` function pointers
//! (`pyrefly::lsp::non_wasm::refinement_hooks`) and installs them via
//! `register_refinedpy_hooks`, so pyrefly's library never depends on
//! this crate or on the `refinedpy` engine it wraps.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

use lsp_types::Diagnostic;
use lsp_types::DiagnosticSeverity;
use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::NumberOrString;
use pyrefly::lsp::non_wasm::refinement_hooks::RefinementHooks;
use pyrefly::lsp::non_wasm::refinement_hooks::register;
use pyrefly::state::state::Transaction;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use refined_kernel::kernel_bridge::kernel_if_loaded;
use refined_kernel::kernel_bridge::load_kernel;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::format_for_hover::format_for_hover;
use refined_sets::format_for_hover::replaces_host_type;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use refinedpy::check::findings_for_module;
use refinedpy::check::findings_for_module_at;
use refinedpy::check::refined_set_at_position;
use refinedpy::cross_module::disk_resolver;
use refinedpy::diagnostic_sentences::stale_marker_refusal;
use refinedpy::fact_export::cached_hash_matches;
use refinedpy::fact_export::export_module;
use refinedpy::fact_export::has_exportable_defs;
use refinedpy::foreign_edge_artifact::cache_artifact_path;
use refinedpy::markers::Marker;
use refinedpy::markers::line_col;
use refinedpy::markers::line_starts_of;
use refinedpy::markers::markers_of;

/// Installs all four RefinedPy hook implementations into pyrefly's
/// registry. Callable once, before serving any request — the served
/// binary's own `main` calls this before `lsp_loop` runs.
pub fn register_refinedpy_hooks() {
    register(RefinementHooks {
        configure_kernel_dylib,
        append_refinedpy_diagnostics,
        export_fact_on_save,
        splice_refinedpy_hover,
    });
}

/// Resolved once before the LSP loop serves requests; `None` when no
/// kernel artifact could be found, in which case every check declines.
static KERNEL_DYLIB: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Resolve and remember the kernel dylib path. Called once from
/// `lsp_loop` before the event loop starts, so every check that runs
/// sees the same answer.
fn configure_kernel_dylib() {
    KERNEL_DYLIB.get_or_init(refinedpy::kernel_path::resolve_kernel_dylib);
}

/// The configured kernel dylib path, if one was found.
fn kernel_dylib() -> Option<&'static PathBuf> {
    KERNEL_DYLIB.get().and_then(|found| found.as_ref())
}

/// The one loaded kernel this server asks. `load_kernel` adopts a
/// process-wide singleton, so retries after a successful load are
/// cache hits; a missing artifact means every check declines.
fn kernel() -> Option<Arc<RefinedTSKernel>> {
    if let Some(loaded) = kernel_if_loaded() {
        return Some(loaded);
    }
    let loaded = load_kernel(kernel_dylib()?).ok()?;
    // The one-time seam installation (a OnceLock — repeat calls are
    // no-ops): crates below the kernel dependency, refined_domain's
    // join-time no-scalar-reread gate today, receive their kernel asks
    // as injected closures routed through the engine's catch discipline.
    refinedpy::kernel_ask::install_kernel_seams(&loaded);
    Some(loaded)
}

/// Append RefinedPy refinement diagnostics for one open handle. Both
/// diagnostic paths (pull and push) reach this through
/// `append_ide_specific_diagnostics`, so this is the one place
/// refinement findings enter the LSP surface. A missing kernel
/// artifact or a non-`.py` handle appends nothing.
fn append_refinedpy_diagnostics(
    transaction: &Transaction<'_>,
    handle: &Handle,
    items: &mut Vec<Diagnostic>,
) {
    if !handle
        .path()
        .as_path()
        .extension()
        .is_some_and(|ext| ext == "py")
    {
        return;
    }
    let Some(kernel) = kernel() else {
        return;
    };
    let Some(ast) = transaction.get_ast(handle) else {
        return;
    };
    let Some(module_info) = transaction.get_module_info(handle) else {
        return;
    };
    // Imports resolve from the open file's own directory, the same way
    // the check CLI resolves them — sibling files are read from disk,
    // so an unsaved sibling buffer contributes its on-disk contents.
    let findings = match handle.path().as_path().parent() {
        Some(directory) => {
            let resolver = disk_resolver(directory.to_path_buf());
            findings_for_module_at(&ast, &resolver, &kernel, Some(directory))
        }
        None => findings_for_module(&ast, &kernel),
    };
    // A fire on a line named by a `# refinedpy: expect-error` marker is
    // an expectation held — the check CLI's own matching rule, via the
    // one shared `Marker::covers` predicate — so the editor stays
    // silent on it. RTS7002 always shows: it says nothing was
    // determined, which no marker (coded or not) is allowed to swallow.
    let source = module_info.contents();
    let markers = markers_of(source);
    let line_starts = line_starts_of(source);
    // Tracks which markers held a real fire, the same way the check
    // CLI's own `matched_markers` vec does — a marker never matched by
    // the loop below is stale, and D2 (marker-parity.md) says a stale
    // marker earns its own RTS7005 diagnostic in the editor, exactly as
    // the CLI already prints-and-fails it (refinedpy_check.rs).
    let mut marker_matched = vec![false; markers.len()];
    for finding in findings {
        let (line, _) = line_col(&line_starts, usize::from(finding.range.start()));
        let matched = markers
            .iter()
            .enumerate()
            .find(|(_, marker)| marker.expected_line == line && marker.covers(finding.code));
        if let Some((index, _)) = matched {
            marker_matched[index] = true;
            continue;
        }
        items.push(Diagnostic {
            range: module_info.to_lsp_range(finding.range),
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String(finding.code.to_owned())),
            code_description: None,
            source: Some("refinedpy".to_owned()),
            message: finding.message.into(),
            related_information: None,
            tags: None,
            data: None,
        });
    }
    for diagnostic in stale_marker_diagnostics(&markers, &marker_matched, &line_starts, source, &module_info) {
        items.push(diagnostic);
    }
}

/// Splices RefinedPy's own refinement spelling into pyrefly's hover, the
/// way `append_refinedpy_diagnostics` splices findings into pyrefly's
/// diagnostics — additive, run after the host's own hover already
/// computed `hover`, never blocking it. A missing kernel artifact, a
/// non-`.py` handle, an unparsed module, or a position with nothing to
/// say (`refined_set_at_position` returns `None`, or the set states
/// nothing past the host type — `format_for_hover` returns `None`)
/// leaves `hover` exactly as the host built it.
fn splice_refinedpy_hover(transaction: &Transaction<'_>, handle: &Handle, position: TextSize, hover: &mut Hover) {
    if !handle
        .path()
        .as_path()
        .extension()
        .is_some_and(|ext| ext == "py")
    {
        return;
    }
    let Some(kernel) = kernel() else {
        return;
    };
    let Some(ast) = transaction.get_ast(handle) else {
        return;
    };
    let Some(set) = (match handle.path().as_path().parent() {
        Some(directory) => {
            let resolver = disk_resolver(directory.to_path_buf());
            refined_set_at_position(&ast, &resolver, &kernel, position)
        }
        None => {
            let no_imports: refinedpy::cross_module::ModuleResolver = &|_: &str| None;
            refined_set_at_position(&ast, no_imports, &kernel, position)
        }
    }) else {
        return;
    };
    let Some(spelled) = format_for_hover(&set) else {
        return;
    };
    splice_spelling_into_hover(hover, &spelled);
}

/// Applies the plugin's own rendering rule (refined-ts-go's own
/// `spliceRefinementSpelling`, `ls/hover_refinedts.go:42-61`) to
/// pyrefly's Markdown hover: a spelling that OPENS WITH A BRACE is a
/// suffix — appended after the fenced type line; anything else REPLACES
/// the right-hand side, everything in that line after the last `=` or
/// `:`. The search is scoped to the FIRST fenced code block's own first
/// line (`HoverValue::format`'s own shape, wasm/hover.rs: "```python\n
/// {kind}{name}{type}\n```..."), never the whole Markdown value — a
/// docstring or parameter-doc section below the fence may itself
/// contain `:` or `=` that must not be mistaken for the type line's own
/// separator.
fn splice_spelling_into_hover(hover: &mut Hover, spelled: &str) {
    let HoverContents::Markup(MarkupContent { value, kind }) = &mut hover.contents else {
        return;
    };
    if !matches!(kind, MarkupKind::Markdown) {
        return;
    }
    let Some(fence_start) = value.find("```") else {
        return;
    };
    let Some(line_start) = value[fence_start..].find('\n').map(|offset| fence_start + offset + 1) else {
        return;
    };
    let Some(line_end) = value[line_start..].find('\n').map(|offset| line_start + offset) else {
        return;
    };
    let type_line = &value[line_start..line_end];
    let spliced = splice_type_line(type_line, spelled);
    value.replace_range(line_start..line_end, &spliced);
}

/// The pure splice itself, over one already-isolated type line — kept
/// apart from `splice_spelling_into_hover` so it is testable without a
/// whole `Hover` response (mirrors the Go twin's own
/// `spliceRefinementSpelling(quickInfo, spelled string) string`).
fn splice_type_line(type_line: &str, spelled: &str) -> String {
    if !replaces_host_type(spelled) {
        return format!("{type_line} {spelled}");
    }
    match type_line.rfind(['=', ':']) {
        Some(cut) => format!("{} {}", &type_line[..=cut], spelled),
        None => format!("{type_line} {spelled}"),
    }
}

/// One RTS7005 diagnostic per marker `marker_matched` names as never
/// holding a real fire — the editor twin of the Go host's `EditorView`
/// (expect_error.go): a marker covering a line nothing fired on is as
/// visible in the editor as a real fire, anchored at the marker's own
/// line rather than the (absent) fire it expected. The anchor is built
/// as a `TextRange` over the marker's own line and rendered through
/// `module_info.to_lsp_range` — the exact seam the fire diagnostics
/// use — so a multi-byte line still reports the right UTF-16 columns.
fn stale_marker_diagnostics(
    markers: &[Marker],
    marker_matched: &[bool],
    line_starts: &[usize],
    source: &str,
    module_info: &Module,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for (index, marker) in markers.iter().enumerate() {
        if marker_matched[index] {
            continue;
        }
        let range = marker_line_text_range(line_starts, source, marker.marker_line);
        out.push(Diagnostic {
            range: module_info.to_lsp_range(range),
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String("RTS7005".to_owned())),
            code_description: None,
            source: Some("refinedpy".to_owned()),
            message: stale_marker_message(marker).into(),
            related_information: None,
            tags: None,
            data: None,
        });
    }
    out
}

/// The stale-marker sentence: names the line the marker expected a
/// fire on and, when the marker carries a reason, prints it. Minted by
/// `diagnostic_sentences::stale_marker_refusal` — THE wording module
/// this checker builds every refinement sentence through — rather than
/// inline here, so a wording fix lands in the one place every other
/// sentence already lives.
fn stale_marker_message(marker: &Marker) -> String {
    let reason = if marker.reason.is_empty() {
        None
    } else {
        Some(marker.reason.as_str())
    };
    stale_marker_refusal(marker.expected_line, reason)
}

/// `line`'s own byte range (1-based) as a `TextRange`, from the same
/// `line_starts` table `line_col` reads. Kept in raw bytes — the
/// UTF-16 conversion for the LSP wire happens once, in
/// `module_info.to_lsp_range`, the same as every fire diagnostic.
fn marker_line_text_range(line_starts: &[usize], source: &str, line: usize) -> TextRange {
    let start_offset = line_starts.get(line - 1).copied().unwrap_or(0);
    let end_offset = line_starts.get(line).map(|next| next - 1).unwrap_or(source.len());
    TextRange::new(
        TextSize::try_from(start_offset).unwrap_or_default(),
        TextSize::try_from(end_offset.max(start_offset)).unwrap_or_default(),
    )
}

/// Save-time fact export (fact-freshness.md, "Python side"): writes
/// `path`'s fact artifact into the project cache so a cross-language
/// caller reading it on the next check sees this save, not the last
/// one. Runs entirely IN-PROCESS — the LSP already holds the loaded
/// kernel (`kernel()` above) — and reads DISK bytes, never the editor
/// overlay: the Go consumer's own spawn reads disk too, so both the
/// producer here and every consumer agree on what "the file" means.
///
/// Three early-outs before a full walk ever runs, in order:
/// - no kernel loaded (every check already declines here; `did_save`
///   still fires, but there is nothing this call could export);
/// - the file does not parse, or the annotated-def gate
///   (`has_exportable_defs`) finds no exportable def —
///   a module with nothing to export costs one shallow scan, not a
///   full `export_module` walk;
/// - the cached artifact's own `target.contentHash` already matches
///   this save's sha256 — the content-hash short-circuit, so an
///   unrelated save (whitespace outside a def, a comment) does not
///   re-walk and re-write a file whose facts have not changed.
///
/// The write is atomic (temp file in the same directory, then
/// rename) — the same discipline `export_file` in the CLI already
/// uses, load-bearing once a live LSP writes the cache mid-session
/// (fact-freshness.md, "Two writers, one entry").
fn export_fact_on_save(path: &Path) {
    if path.extension().is_none_or(|ext| ext != "py") {
        return;
    }
    let Some(kernel) = kernel() else {
        return;
    };
    let Ok(source) = std::fs::read(path) else {
        return;
    };
    let Ok(text) = std::str::from_utf8(&source) else {
        return;
    };
    let Ok(parsed) = ruff_python_parser::parse_module(text) else {
        return;
    };
    let module = parsed.into_syntax();
    if !has_exportable_defs(&module) {
        return;
    }
    let artifact_path = cache_artifact_path(&path.to_string_lossy());
    if cached_hash_matches(&artifact_path, &source) {
        return;
    }
    let entry_directory = path.parent().filter(|dir| !dir.as_os_str().is_empty());
    let resolver = disk_resolver(entry_directory.unwrap_or(Path::new(".")).to_path_buf());
    let basename = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned());
    let export = export_module(&module, &source, &basename, &resolver, &kernel, entry_directory);
    for omission in &export.omissions {
        eprintln!(
            "{}: '{}' is not exported: {}",
            path.display(),
            omission.function,
            omission.reason
        );
    }
    let Ok(rendered) = serde_json::to_string_pretty(&export.artifact) else {
        return;
    };
    write_artifact_atomically(&artifact_path, &rendered);
}

/// Writes `rendered` to `artifact_path` atomically: a temp file in the
/// SAME directory, then a rename — atomic on the same volume, and the
/// same discipline the CLI's own `export_file` uses so a reader on the
/// other side of the edge never observes a torn write, whichever
/// writer runs last (fact-freshness.md, "Two writers, one entry").
fn write_artifact_atomically(artifact_path: &Path, rendered: &str) {
    let Some(parent) = artifact_path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let file_name = artifact_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact".to_owned());
    let temp_path = artifact_path.with_file_name(format!("{file_name}.tmp.{}", std::process::id()));
    if std::fs::write(&temp_path, format!("{rendered}\n")).is_err() {
        return;
    }
    if std::fs::rename(&temp_path, artifact_path).is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;

    use super::*;

    /// A `{...}`-shaped spelling APPENDS after the type line — the
    /// plugin's own `replaces_host_type` rule (a brace opens a suffix),
    /// mirrored from refined-ts-go's own `spliceRefinementSpelling`
    /// (`ls/hover_refinedts.go:42-61`).
    #[test]
    fn a_brace_spelling_appends_after_the_type_line() {
        let spliced = splice_type_line("samples: list[float]", "{len >= 1}");
        assert_eq!(spliced, "samples: list[float] {len >= 1}");
    }

    /// A spelling that does NOT open with a brace (a single value, like
    /// `format_for_hover`'s own "one value renders as that value"
    /// convention) REPLACES the type line's own right-hand side —
    /// everything after the last `=`/`:` — rather than doubling it.
    #[test]
    fn a_non_brace_spelling_replaces_the_line_past_the_last_separator() {
        let spliced = splice_type_line("age: int", "1");
        assert_eq!(spliced, "age: 1");
    }

    /// A type line with no `=`/`:` at all (a bare type with no name
    /// prefix, e.g. hovering a type alias' own definition) has no
    /// right-hand side to cut — the spelling appends instead of being
    /// silently dropped.
    #[test]
    fn a_non_brace_spelling_appends_when_the_line_has_no_separator() {
        let spliced = splice_type_line("float", "1");
        assert_eq!(spliced, "float 1");
    }

    /// The whole-`Hover` splice is scoped to the FIRST fenced code
    /// block's own first line — a docstring line below the fence
    /// containing its own `:` must not be mistaken for the type line.
    #[test]
    fn the_hover_splice_only_touches_the_fenced_type_line_not_the_docstring() {
        let mut hover = Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: "```python\nsamples: list[float]\n```\n---\nNote: length matters here.".to_owned(),
            }),
            range: None,
        };
        splice_spelling_into_hover(&mut hover, "{len >= 1}");
        let HoverContents::Markup(MarkupContent { value, .. }) = &hover.contents else {
            panic!("expected Markup contents");
        };
        assert_eq!(
            value,
            "```python\nsamples: list[float] {len >= 1}\n```\n---\nNote: length matters here."
        );
    }

    /// The dylib the tests below ask, the same load path
    /// `check.rs`'s own `loaded_kernel` uses — `None` (rather than a
    /// panic) when the native artifact has not been built, so a
    /// checkout without `pnpm kernel:native` run yet skips these tests
    /// instead of failing them.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = refined_kernel::kernel_bridge::dylib_path();
        if !refined_kernel::kernel_bridge::kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn parsed(source: &str) -> ruff_python_ast::ModModule {
        ruff_python_parser::parse_module(source)
            .expect("fixture source parses")
            .into_syntax()
    }

    fn no_imports_resolver() -> refinedpy::cross_module::ModuleResolver<'static> {
        &|_: &str| None
    }

    /// A `Hover` shaped exactly like `HoverValue::format`'s own
    /// post-trim output: the fenced host type line, and nothing past
    /// it (no "Go to" link, no "Type source" block) — the same shape
    /// `splice_refinedpy_hover` receives from `get_hover_with_verbosity`
    /// before it appends the refined-set line.
    fn host_only_hover(type_line: &str) -> Hover {
        Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!("```python\n{type_line}\n```"),
            }),
            range: None,
        }
    }

    /// The whole splice, end to end: a real declared set — `Level`'s
    /// own `0.0 <= x <= 1.0` bound, read at the parameter annotation's
    /// own position — spliced onto a host-only hover produces exactly
    /// two lines inside the fence: the host type, then the refined
    /// set. This is the shape a real editor hover shows once
    /// `HoverValue::format` (lib/lsp/wasm/hover.rs) has dropped the
    /// stock "Go to"/"Type source" blocks and
    /// `splice_refinedpy_hover` (this file) has appended the set line.
    #[test]
    fn a_declared_set_splices_a_second_line_after_the_host_type() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "\n",
            "def f(level: Level) -> None:\n",
            "    pass\n",
        );
        let module = parsed(source);
        let position = ruff_text_size::TextSize::try_from(source.find("Level) -> None").unwrap())
            .unwrap();
        let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
            .unwrap_or_else(|| panic!("expected a declared set at the parameter annotation"));
        let spelled =
            format_for_hover(&set).unwrap_or_else(|| panic!("expected a hover spelling for 0.0..1.0"));
        let mut hover = host_only_hover("level: float");
        splice_spelling_into_hover(&mut hover, &spelled);
        let HoverContents::Markup(MarkupContent { value, .. }) = &hover.contents else {
            panic!("expected Markup contents");
        };
        let lines: Vec<&str> = value.lines().collect();
        assert_eq!(
            lines,
            vec!["```python", &format!("level: float {spelled}"), "```"],
            "expected exactly the host type line followed by the refined-set line, got: {value}"
        );
    }

    /// A position with no refinement vocabulary in scope
    /// (`refined_set_at_position` returns `None` — no `type` alias, no
    /// `Annotated`/`Literal` import anywhere in the module) leaves the
    /// hover exactly as the host built it: the host type line alone,
    /// no second line, no junk block.
    #[test]
    fn a_position_with_no_declared_vocabulary_leaves_the_host_line_alone() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = "def f(level: float) -> None:\n    pass\n";
        let module = parsed(source);
        let position = ruff_text_size::TextSize::try_from(source.find("float) -> None").unwrap())
            .unwrap();
        assert!(
            refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_none(),
            "a module with no refinement vocabulary must answer no set at any position"
        );
        // `splice_refinedpy_hover` takes exactly this early-out (its own
        // `let Some(set) = ... else { return }`) whenever
        // `refined_set_at_position` answers `None` — no splice call
        // happens, so the host-built hover is untouched. Asserted here
        // directly on the value a real hover would carry at this
        // position: the fenced host type line, nothing more.
        let hover = host_only_hover("level: float");
        let HoverContents::Markup(MarkupContent { value, .. }) = &hover.contents else {
            panic!("expected Markup contents");
        };
        assert_eq!(value, "```python\nlevel: float\n```");
    }

    /// `total`'s own position in `audio_level_unclamped.py`
    /// (`total = sum(s * s for s in samples)`) now ANSWERS a set — the
    /// relational-sum walk records an evaluated node at the Assign
    /// statement's range before binding (check.rs walk_relational_sum,
    /// 2026-08-22), so `refined_set_at_position` serves the derived
    /// total and the hover gains its second line. Old premise, cited:
    /// this pin asserted `is_none` while the recognized binding
    /// bypassed the recorder.
    #[test]
    fn the_measured_total_position_answers_the_derived_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "import math\n",
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
            "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "\n",
            "def audio_level_unclamped(samples: Annotated[list[Sample], Field(min_length=1)]) -> Level:\n",
            "    total = sum(s * s for s in samples)\n",
            "    return math.sqrt(total / len(samples))\n",
        );
        let module = parsed(source);
        let position =
            ruff_text_size::TextSize::try_from(source.find("total = sum").unwrap()).unwrap();
        assert!(
            refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_some(),
            "total's own position answers the derived set now that the relational-sum walk records its evaluated node"
        );
    }

    /// A bare in-memory `Module` over `source` — enough for
    /// `to_lsp_range` to translate a `TextRange` into an LSP position,
    /// with no LSP transaction or open file involved.
    fn test_module_info(source: &str) -> Module {
        Module::new(
            ModuleName::from_str("test_module"),
            ModulePath::filesystem(PathBuf::from("test_module.py")),
            Arc::new(source.to_owned()),
        )
    }

    /// A code-less marker holds any real fire on its expected line, and
    /// a marker that holds nothing earns its own 7005 — the stale-marker
    /// half of D2 (marker-parity.md), tested at the function that
    /// decides it rather than through the LSP.
    #[test]
    fn a_marker_matching_no_finding_earns_a_stale_diagnostic() {
        let source = "over = 200\n";
        let markers = markers_of("# refinedpy: expect-error\nover = 200\n");
        let line_starts = line_starts_of(source);
        // No finding ever matched this marker (the caller's own loop
        // decides that; here it is simulated directly).
        let marker_matched = vec![false; markers.len()];
        let diagnostics = stale_marker_diagnostics(
            &markers,
            &marker_matched,
            &line_starts,
            source,
            &test_module_info(source),
        );
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code, Some(NumberOrString::String("RTS7005".to_owned())));
    }

    /// A marker recorded as matched produces no diagnostic.
    #[test]
    fn a_matched_marker_earns_no_diagnostic() {
        let source = "over = 200\n";
        let markers = markers_of("# refinedpy: expect-error\nover = 200\n");
        let line_starts = line_starts_of(source);
        let marker_matched = vec![true; markers.len()];
        let diagnostics = stale_marker_diagnostics(
            &markers,
            &marker_matched,
            &line_starts,
            source,
            &test_module_info(source),
        );
        assert!(diagnostics.is_empty());
    }

    /// `stale_marker_message` forwards the marker's own expected line
    /// and reason into `diagnostic_sentences::stale_marker_refusal`
    /// unchanged — the wording itself is that function's own contract,
    /// pinned by its own tests, not restated here.
    #[test]
    fn the_stale_message_forwards_the_expected_line_and_reason() {
        let markers = markers_of("# refinedpy: expect-error — 200 is outside the set\nover = 200\n");
        let message = stale_marker_message(&markers[0]);
        assert_eq!(message, stale_marker_refusal(2, Some("200 is outside the set")));

        let markers_no_reason = markers_of("# refinedpy: expect-error\nover = 200\n");
        let message_no_reason = stale_marker_message(&markers_no_reason[0]);
        assert_eq!(message_no_reason, stale_marker_refusal(2, None));
    }

    /// The atomic write lands the exact bytes at the final path and
    /// leaves no `.tmp.<pid>` file behind.
    #[test]
    fn the_atomic_write_lands_the_final_file_and_no_temp_remains() {
        let dir = std::env::temp_dir().join(format!(
            "refinedpy_export_on_save_test_{}_atomic_write",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let artifact_path = dir.join("m.py.refined.json");
        write_artifact_atomically(&artifact_path, "{\"ok\": true}");
        let written = std::fs::read_to_string(&artifact_path).expect("artifact written");
        assert!(written.contains("\"ok\": true"), "{written}");
        let temp_path = dir.join(format!("m.py.refined.json.tmp.{}", std::process::id()));
        assert!(!temp_path.exists(), "the temp file must be renamed away, not left behind");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
