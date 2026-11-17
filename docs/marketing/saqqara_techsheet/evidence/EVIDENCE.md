# Saqqara tech sheet: measured evidence

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Every figure below comes from a run listed here, and every table is the output of `summarize.py` over
a results file in this folder, pasted unedited. Proof figures come only from public files: files tracked in this repository, plus three synthetic fixtures made by the repository's
own generators, each labelled SYNTHETIC wherever it appears.

## Status (public edition, 2026-11-17)

- **Measured:**
  - M1 (x86-64);
  - M3, the test suite, at `20fdf7a` (§5) and at `b7eb0a7` (§5.1);
  - M5, the cover figure (§6);
  - the anchor contrast (§7);
  - M2a, the same code and bytes on two architectures (§8);
  - M2b, the pinned environment (§9);
  - the default reader's text positions before and after their fix, merged as `b7eb0a7` (§10 to §10.3);
  - M4, Docling 2.150.1 beside Saqqara (§11).
- **Downloads:** every download behind §9 and §11 was approved from an itemised list before it was fetched.
  The lists and pip's own records are in this folder. §11 reports the one model repository fetched outside
  its list.
- **Which commit:** the Saqqara figures of §10 and §11 describe `b7eb0a7` (§10.3).
- **Not measured yet:** public Word, Excel and PowerPoint files.
- **Not cited by the sheet:** the SYNTHETIC fixtures and the contrast PDF.
- **What this edition changes:** host names and addresses, local paths and the name of the measurement
  environment are replaced by neutral labels (the x86-64 laptop, the ARM64 workstation, `<bench>`,
  `<env>`, `work_env_…`). pip's install reports are reduced to package, version, URL and hash. Every
  measured value is unchanged.

## 0. Files in this folder

| file | what it is |
|---|---|
| `stage_inputs.py` | copies the public inputs (refusing any file `git ls-files` does not list), generates the three synthetic fixtures, writes `inputs/MANIFEST.json` |
| `inputs_manifest.json` | that manifest: id, relative path, bytes, sha256, origin of every input |
| `bench_saqqara.py` | the measurement, in four parts, each in its own process: `kernel`, `objects`, `explorer`, `cli` |
| `summarize.py` | prints the tables of this file from one or two results files |
| `results_m1_x86_64.json` | M1, on the x86-64 laptop |
| `m3_saqqara_pytest.log`, `m3_others_pytest.log` | M3, the two pytest outputs, the worktree path replaced by `<worktree>` |
| `m3_saqqara_pytest_b7eb0a7.log`, `m3_others_pytest_b7eb0a7.log` | §5.1, the suite at the merged commit `b7eb0a7` |
| `hero_figure.py`, `hero_p1.png`, `hero_p3.png`, `hero_p1.json`, `hero_p3.json` | M5, the cover figure and the boxes drawn on it |
| `inputs_manifest_m2.json` | the inputs of M2a: the same five public files, the three synthetic ones regenerated (§8) |
| `results_m2_x86_64.json`, `results_m2_aarch64.json` | M2a, the same bench and the same input bytes on both hosts |
| `m2b_plan_downloads.py`, `m2b_planned_downloads.json` | M2b, planned only: the wheels a fresh install of `ragix[saqqara,explorer]` would fetch on the ARM64 workstation, from PyPI's metadata, 2026-11-16 21:57:54 (+01:00, the time the list was written); nothing downloaded. odfpy 1.4.1 has no wheel (source archive, 717 045 bytes; building it fetches setuptools) |
| `m2b_requirements_pinned.txt` | M2b: the 20 packages installed, each pinned by version and sha256 |
| `m2b_install.log`, `m2b_install_report.json`, `m2b_pip_freeze.txt` | M2b: pip's log (bench path replaced by `<bench>`), pip's install report (file, URL and hash of every download), and the environment's freeze |
| `results_m2b_aarch64_pinned.json` | M2b: the bench in the pinned environment on the ARM64 workstation |
| `compare_results.py`, `tree_diff.py`, `m2b_tree_diff_pymupdf.txt` | field-by-field comparison of two results; node-by-node comparison of two trees, and its output for M2a against M2b |
| `extend_inputs.py`, `inputs_manifest_1b.json` | §10: M2a's input bytes plus the Ghostscript contrast PDF |
| `results_1b_{before,after}_{x86_64,aarch64,aarch64_pinned}.json` | §10: 20fdf7a against d859c87, on x86 and on the ARM64 workstation in both environments |
| `results_1b_after3_{x86_64,aarch64,aarch64_pinned}.json` | §10.2: the pushed head 1237905, "after" side only |
| `results_1b_after2_{x86_64,aarch64,aarch64_pinned}.json` | §10.1: the review commit 2ddf2af, "after" side only |
| `before_after.py`, `1b_before_after_pinned.md`, `1b_tree_diff_pinned.txt` | §10: per-PDF measures before and after with the roots, and the node-by-node tree comparison |
| `m4_plan_downloads.py`, `m4_planned_downloads.json` | M4, the plan approved before any download: wheels and model files from public metadata, nothing downloaded |
| `m4_requirements_torch_cpu.txt`, `m4_requirements_opencv.txt` | M4, the wheels installed pinned by sha256 |
| `m4_install_report_{torch_cpu,pypi,opencv}.json`, `m4_install_*.log`, `m4_pip_freeze.txt`, `m4_download_sizes.py`, `m4_downloads.json` | M4, what pip fetched (file, URL, hash), with sizes from the indexes; logs with the bench path replaced by `<bench>` |
| `m4_models_download.log`, `m4_models_files.txt` | M4, the model files fetched, with sizes, and the disk used |
| `docling_bench.py`, `results_m4_docling_aarch64.json` | M4, the Docling measurement |
| `docling_json_diff.py`, `m4_docling_json_diff.txt` | M4, what differs between Docling's JSON exports of the same input |
| `saqqara_rss.py`, `saqqara_rss_pypdf.json`, `saqqara_rss_pymupdf.json` | M4, Saqqara's peak memory, one process per file |
| `side_by_side.py`, `m4_side_by_side.md` | M4, Saqqara and Docling per input |
| `docling_markdown_techsheet_pdf.md` | M4, Docling's Markdown export of the tech sheet, as an illustration only |
| `anchor_check.py`, `anchors_non_chromium.json` | §7, text anchors outside the page on a PDF not printed by Chromium |

## 1. Inputs

| id | path | bytes | origin |
|---|---|---|---|
| techsheet_pdf | `docs/marketing/ragix_techsheet/Adservio-RAGIXTechSheet.pdf` | 2 007 073 | public, tracked; A4, 19 pages, printed by Chromium (Skia/PDF) |
| slides_pdf | `docs/marketing/ragix_techsheet/ragix-tech-sheet/RAGIX-Tech-Sheet-Slides.pdf` | 2 676 406 | public, tracked; 14 pages of 1280×720 pt, one picture per page, no text layer |
| sovereign_pdf | `docs/SOVEREIGN_LLM_OPERATIONS.pdf` | 1 155 159 | public, tracked; A4, 38 pages, Typora/Electron |
| saqqara_readme_md | `ragix_kernels/saqqara/README.md` | 8 149 | public, tracked |
| koas_saqqara_md | `docs/KOAS_SAQQARA.md` | 52 678 | public, tracked |
| twin_grid_docx | generated: `tests/saqqara/generators.py:twin_grid_docx` | 36 820 | SYNTHETIC |
| twin_grid_xlsx | generated: `tests/saqqara/generators.py:twin_grid_xlsx` | 4 967 | SYNTHETIC |
| twin_grid_pptx | generated: `tests/saqqara/generators.py:twin_grid_pptx` | 28 487 | SYNTHETIC |

The repository tracks no `.docx`, `.xlsx` or `.pptx` file (`git ls-files` at 20fdf7a), hence the three
synthetic fixtures: one small table with a column span and a vertical merge, written as a word-processing
document, a spreadsheet and a presentation. The page counts and producers above are from `pdfinfo`; the
sha256 of each input is in `inputs_manifest.json`. Both text PDFs set their text mostly in Type 3 fonts
(`pdffonts`: 41 of 46 font entries are Type 3 for the tech sheet, 40 of 66 for the operations document).

## 2. How to reproduce M1

```bash
W=<this worktree>                       # RAGIX at 20fdf7a
E=$W/docs/marketing/saqqara_techsheet/evidence
B=<an empty bench directory>
conda run -n <env> env PYTHONPATH=$W python $E/stage_inputs.py --repo $W --bench $B
cd $B && conda run -n <env> env PYTHONPATH=$W python $E/bench_saqqara.py --out $E/results_m1_x86_64.json --work work_m1
cd $E && python3 summarize.py results_m1_x86_64.json
```

Environment: the measurement environment, a conda env with Python 3.12.13, pypdf 6.17.0, PyMuPDF 1.28.0, pypdfium2 5.12.1 with
pdfium 152.0.7947.0, python-docx 1.2.0, openpyxl 3.1.5, python-pptx 1.0.2, SQLite 3.53.3), RAGIX imported
from this worktree at 20fdf7a with no change under `ragix_kernels/`. The results record a digest of the
319 Python files of `ragix_kernels` as imported, `ec3c65f45482030d…`, so a second host can show that it
ran the same code. CPU, from `lscpu`: 12th Gen Intel Core i9-12900H, 20 logical CPUs. M1 ran on
2026-11-16 from 18:48:28 to 18:50:22 (+01:00); the one-minute load average was 2.34 at its start and 3.74 at its end.
Wall times are single-process and not pinned to a core.

What the bench runs, and on which surface:

- **kernel**: `SaqqaraKernel().run(KernelInput(workspace=…, config={"source": {"path": …}}))` on each
  file, twice, with the default pdf text reader (pypdf) and with the opt-in one (`"pdf": {"text_reader":
  "pymupdf"}`), and once more with `promote_outline: true`. Then the whole `inputs/` folder twice per
  reader. Then the tech sheet read again from another path under another name.
- **objects**: the K6 passes, which exist only as a library (`docs/KOAS_SAQQARA.md` §3.3): the reader
  given an asset store (pictures), `VectorRegionAnalyzer` (drawn regions), `CaptionBindingAnalyzer`,
  `FormatHeadingsAnalyzer` (headings shown by size or weight). A run of the kernel produces none of these.
- **explorer**: `saqqara.explorer.digest_pdf` (PyMuPDF; native table observations with a box for every
  cell), then `explore()`.
- **cli**: `saqqaractl run inputs`, `index`, and `search` for two queries, with no embedder configured
  (lexical lane only).

## 3. M1 results, x86

Host: x86_64, 12th Gen Intel(R) Core(TM) i9-12900H, 20 logical CPUs, Python 3.12.13, code digest ec3c65f45482030d…, 2 runs per file.

| input | reader | pages | nodes | kinds | anchors off page | tables (rows×cols, cells) | headings (+outline) | sections accepted / candidates | abstentions | wall s (run 1 / 2) | s per page | roots equal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| techsheet_pdf | pymupdf | 19 | 882 | page 19, paragraph 862 | 0 / 862 | 0 | 0 (+35) | 49 / 73 | 0 | 1.132 / 1.101 | 0.060 / 0.058 | yes |
| techsheet_pdf | pypdf | 19 | 894 | page 19, paragraph 874 | 650 / 874 | 0 | 0 (+35) | 49 / 73 | 0 | 1.944 / 1.898 | 0.102 / 0.100 | yes |
| slides_pdf | pymupdf | 14 | 15 | page 14 | – | 0 | 0 (+0) | 0 / 0 | saqqara.text_layer:no-text-layer 14 | 0.018 / 0.018 | 0.001 / 0.001 | yes |
| slides_pdf | pypdf | 14 | 15 | page 14 | – | 0 | 0 (+0) | 0 / 0 | saqqara.text_layer:no-text-layer 14 | 0.018 / 0.017 | 0.001 / 0.001 | yes |
| sovereign_pdf | pymupdf | 38 | 14370 | heading 117, page 38, paragraph 14214 | 0 / 14214 | 0 | 117 (+24) | 333 / 719 | 0 | 4.504 / 3.947 | 0.118 / 0.104 | yes |
| sovereign_pdf | pypdf | 38 | 14857 | heading 117, page 38, paragraph 14701 | 13068 / 14701 | 0 | 117 (+24) | 370 / 719 | 0 | 5.736 / 6.579 | 0.151 / 0.173 | yes |
| saqqara_readme_md | default | – | 37 | heading 10, paragraph 26 | – | 0 | 10 (+0) | 0 / 0 | 0 | 0.002 / 0.002 | – | yes |
| koas_saqqara_md | default | – | 204 | heading 41, paragraph 162 | – | 0 | 41 (+0) | 0 / 0 | 0 | 0.006 / 0.006 | – | yes |
| twin_grid_docx (SYNTHETIC) | default | – | 13 | cell 11, table 1 | – | 4×3, 11 (data-form) | 0 (+0) | 0 / 0 | 0 | 0.033 / 0.007 | – | yes |
| twin_grid_xlsx (SYNTHETIC) | default | – | 13 | cell 10, section 1, table 1 | – | 4×3, 10 (table) | 0 (+0) | 1 / 1 | 0 | 0.048 / 0.004 | – | yes |
| twin_grid_pptx (SYNTHETIC) | default | 1 | 14 | cell 11, slide 1, table 1 | – | 4×3, 11 (data-form) | 0 (+0) | 0 / 0 | 0 | 0.034 / 0.004 | 0.034 / 0.004 | yes |

Whole input set, pymupdf: 8 document(s), 15548 nodes. Refused 1, duplicate 0. 14 abstention(s), 0 drop(s) counted. root 63c2d5b7fd7e. merkle_root 63c2d5b7fd7ebadd526b5e036302351889499e89de9476a9cb9c310a77299c9d, source_root 8428e3c40ad260f01203b94a7e6c1265aeae3bc90e5c90cc7f57f94e77451996, refusals [{'path': 'inputs/MANIFEST.json', 'reason': 'unsupported-format'}], wall s 5.168 / 5.094, roots equal across runs: True.
Whole input set, pypdf: 8 document(s), 16047 nodes. Refused 1, duplicate 0. 14 abstention(s), 0 drop(s) counted. root e7c14b4fc29b. merkle_root e7c14b4fc29bfe08818357c62749ac2f078d1252ff27d9266256f98d968c0eb7, source_root 8428e3c40ad260f01203b94a7e6c1265aeae3bc90e5c90cc7f57f94e77451996, refusals [{'path': 'inputs/MANIFEST.json', 'reason': 'unsupported-format'}], wall s 7.601 / 8.080, roots equal across runs: True.

Same bytes from another path (work_m1/moved/renamed_copy.pdf): merkle_root equal False, source_root equal True, path-free tree equal True.

| input | pictures (assets) | drawn ink: marks / clusters / promoted regions | regions refused or excluded | caption binding: figures / bound / abstained | headings by size: promoted (nodes before → after) |
|---|---|---|---|---|---|
| techsheet_pdf | 4 (4) | 1818 / 82 / 58 | too-few-operators 17, too-small 7, page-furniture 225 | 4 / 1 / no-candidate-within-gap 3 | 73 (0 → 73) |
| slides_pdf | 14 (14) | 0 / 0 / 0 | 0 | 14 / 0 / no-candidate-within-gap 14 | 0 (0 → 0) |
| sovereign_pdf | 0 (0) | 5572 / 38 / 0 | region-spans-page 38 | 0 / 0 / 0 | 0 (117 → 117) |

| input | extractor | pages | text spans | drawings | native tables | rows per table | cells | cells without geometry | digest s | explore s |
|---|---|---|---|---|---|---|---|---|---|---|
| techsheet_pdf | pymupdf 1.28.0 | 19 | 862 | 1434 | 10 | 5, 6, 4, 4, 4, 3, 1, 4, 8, 5 | 59 | 0 | 1.586 | 1.062 |
| slides_pdf | pymupdf 1.28.0 | 14 | 0 | 0 | 0 | – | 0 | 0 | 0.906 | 0.005 |
| sovereign_pdf | pymupdf 1.28.0 | 38 | 14214 | 3737 | 34 | 4, 3, 2, 3, 5, 6, 5, 5, 4, 6, 6, 4, 22, 11, 18, 5, 8, 2, 5, 8, 5, 4, 5, 9, 6, 5, 7, 3, 5, 3, 6, 5, 5, 8 | 796 | 0 | 3.890 | 7.212 |

saqqaractl run: counts {'duplicate': 0, 'read': 8, 'refused': 1}, merkle_root e7c14b4fc29bfe08818357c62749ac2f078d1252ff27d9266256f98d968c0eb7, wall 9.236 s. index: 8 documents, 16106 chunks, 3 objects, 0 edges, dense lane 'disabled (no embedder)', wall 3.304 s.
search 'merkle_root': 5 hits in 0.553 s; top 3: #1 paragraph p.[18] «compute_inputs_merkle_root» (SOVEREIGN_LLM_OPERATIONS.pdf); #2 heading p.[18] «Merkle Root for Pyramidal Synthesis» (SOVEREIGN_LLM_OPERATIONS.pdf); #3 paragraph p.[18] «MERKLE ROOT FOR PYRAMIDAL SYNTHESIS» (SOVEREIGN_LLM_OPERATIONS.pdf)
search 'provenance': 5 hits in 0.560 s; top 3: #1 paragraph p.[3] «PROVENANCE» (Adservio-RAGIXTechSheet.pdf); #2 paragraph p.[9] «provenance» (Adservio-RAGIXTechSheet.pdf); #3 paragraph p.[6] «provenance graph» (Adservio-RAGIXTechSheet.pdf)


## 4. What the M1 figures show, and what they do not

1. **Same input, same roots.** For every file, with both readers, both runs gave the same `merkle_root`
   and the same `source_root`, and so did the whole set. `saqqaractl run` on the whole set gave the same
   `merkle_root` as the kernel called from Python with the same reader (`e7c14b4f…`).
2. **What each root covers.** The same bytes read from another path under another name gave the same
   `source_root`, the same tree once the recorded source path is blanked, and a **different**
   `merkle_root`. `merkle_root` hashes each tree as serialised, and a tree records the path it was read
   from. So a comparison across hosts must read from the same relative paths, which this bench does.
3. **Tables in PDF.** The run kernel builds no `table` node from a PDF: the pdf reader makes pages and
   paragraphs, and the table analyzers type spreadsheet blocks and declared word-processing or slide
   tables. PDF tables come from the Explorer's PyMuPDF digest: 10 native tables on the tech sheet (59
   cells) and 34 on the operations document (796 cells), every cell with a box. These counts are
   PyMuPDF's `find_tables`, kept as observations; they have not been checked against the pages by eye.
4. **No text layer: an abstention, not a guess.** The slides PDF has no text layer. Saqqara records one
   `no-text-layer` abstention per page (14) with the page's signals, reads the 14 pictures when given an
   asset store, and runs no OCR.
5. **The two pdf text readers differ on these files.** With pypdf, 650 of 874 text anchors of the tech
   sheet and 13 068 of 14 701 of the operations document lie outside their page box; with PyMuPDF, 0 of
   862 and 0 of 14 214. Both documents set their text in Type 3 fonts. The cause has not been diagnosed
   here. The PyMuPDF reader was also faster on both (per page, 0.058–0.060 s against 0.100–0.102 s on the
   tech sheet). It is the AGPL-3.0 opt-in (K6.16); pypdf, BSD-3-Clause, is the default.
6. **Headings.** Both Markdown files keep their declared headings (10 and 41). The operations PDF
   carries a declared outline (117 headings, both readers). The tech-sheet PDF has none: the opt-in
   outline pass adds 35 inferred headings from numbered lines, and the library pass on font size
   promotes 73 lines. Neither is part of a default run.
7. **What is refused is counted.** The whole-set run is offered every file in `inputs/`, including
   `MANIFEST.json`; it refuses that file as `unsupported-format` and says so.
8. **Markdown tables are not typed.** The md reader emits headings and paragraphs only. A pipe table
   becomes one `paragraph` node holding its raw lines: 1 such node in the README (11 lines) and 32 in
   the user reference (234 lines).
9. **Drawn regions.** On the tech sheet, 1 818 drawing marks were grouped into 82 clusters; 58 were
   promoted to drawn regions, 17 refused as too few operators, 7 as too small, and 225 marks excluded as
   page furniture. On the operations document, each of the 38 pages yields one cluster spanning the
   page, refused as `region-spans-page`, and nothing is promoted.
10. **Timing scope.** The times are those of one process on a laptop CPU with other work running, two
    runs each, the first including lazy imports (visible on the synthetic files: 0.033 s, then 0.007 s).
    They are orders of magnitude, not a benchmark.

## 5. M3: the Saqqara test suite at 20fdf7a

```bash
cd $W && conda run -n <env> env PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest tests/saqqara -p no:cacheprovider -rs
```

The RAGIX `pyproject.toml` sets no pytest `addopts`, so the summary line prints as is. Read from it
(`m3_saqqara_pytest.log`), 2026-11-16 18:52:54 to 18:54:27:

- collected 1858 items: **1848 passed, 9 skipped, 1 xfailed**, 8 warnings, in 90.75 s.
- The 9 skips, by reason:
  - 3: live embedding test (`OLLAMA_LIVE=1` not set);
  - 3: "this fixture pins an abstention, not a split" (`test_k3_analyzers.py`);
  - 2: "counted as a refusal, asserted in K3.43 and K3.42" (`test_k3hi_sections_outline.py`);
  - 1: "no .git/hooks in this checkout" (`test_k0_spec.py`; this checkout is a git worktree).
- The one xfail: the only xfail markers under `tests/saqqara` are the strict ones of
  `test_explorer_slice4_fields.py`, which state a defect measured on the frozen reader.

The other tests whose files mention saqqara: `tests/harvest`, `tests/tender`,
`tests/sealed/test_extract_readers.py`, `tests/test_kernels.py`, `tests/test_translate_extract.py`, same
command with `-rsx` (`m3_others_pytest.log`).

- collected 773 items: **747 passed, 25 skipped, 1 failed**, 16 warnings, in 4.38 s.
- The failure is `tests/test_kernels.py::TestRequiresResolution::test_every_requirement_resolves_to_exactly_one_kernel`.
  `doc_final_report` requires `doc_summarize_tutored` and `doc_func_extract`, which did not register.
  Importing their modules in this environment fails with `No module named 'httpx'`. It is an environment
  gap, outside saqqara.
- The skips are stores that are not on this host (21), `core_abstracts.jsonl` not in this commit (3), and
  `pymupdf4llm` not installed (1).

### 5.1 The suite at b7eb0a7, the merged commit the sheet describes

The same procedure and environment as §5, in this worktree, now at `b7eb0a7` (tree `e1ec50a7…`, §10.3). The
untracked sheet folder was the only change, before and after (`git status`). The worktree path is replaced
by `<worktree>` in the logs (`m3_saqqara_pytest_b7eb0a7.log`, `m3_others_pytest_b7eb0a7.log`). Run
2026-11-17 07:23:37 to 07:25:12 (+01:00), with `-rsx` so that the xfail is named.

- **`tests/saqqara`:** collected 1860 items: **1850 passed, 9 skipped, 1 xfailed**, 8 warnings, in 85.31 s.
  - Against `20fdf7a` (§5): 2 more tests collected and passed (1858 → 1860; 1848 → 1850), the fix's own.
  - The 9 skips have the same reasons and counts as in §5.
  - The xfail, now named: `test_explorer_slice4_fields.py::test_a9_role_word_lines_are_listed_never_read[mid]`,
    "two leading values derive a bound that closes the window (flagged); repair deferred".
- **The other tests that mention saqqara** (the same five paths as §5): collected 773 items: **747 passed, 25
  skipped, 1 failed**, 16 warnings, in 5.32 s.
  - The same failure as §5, the kernel-registry test (`httpx` missing in this environment).
  - The skip set is identical to §5.

## 6. M5: the cover figure

```bash
cd $B && conda run -n <env> env PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 \
  python $E/hero_figure.py --page 3 --width 1600 --out $E/hero_p3.png --boxes $E/hero_p3.json --work <dir>
```

`hero_p3.png` and `hero_p1.png` are 1600×2459 px: the page at 1600 px wide, plus a legend band.

- **Renderer:** each page is rasterised by Saqqara's default renderer, `default_renderer()` (pypdfium2
  5.12.1, pdfium 152.0.7947.0).
- **Overlay:** the boxes Saqqara records for that page, one colour and one line style per kind:
  - **paragraph** (blue, thin, light fill): the text anchor the reader records, its width, and its font
    size as the height. The reader records no text box, so this rectangle is drawn from those three facts.
    Read with the PyMuPDF text reader, because with pypdf most anchors of this file fall off the page
    (§4.5).
  - **figure** (orange, thick): a picture placement box (the reader given an asset store).
  - **vector_region** (aqua, dashed): a drawn region promoted by the library pass.
- **Counts:**
  - page 3: 56 paragraphs, 0 figures, 2 drawn regions (the four stat cards and the "core question" box);
  - page 1: 18 paragraphs, 2 figures, 1 drawn region.
- **Colours:** slots 1–3 of the reference categorical palette (`#2a78d6`, `#eb6834`, `#1baf7a`), checked
  with the palette validator: all checks pass on a light surface. Aqua is under 3:1 contrast, which is why
  the legend is labelled and the line style differs per kind.
- **Determinism:** two builds of each PNG are byte-identical.
- The `.json` files list every box in PDF points and in pixels, so the figure can be restyled without
  re-reading the PDF.

## 7. Text anchors outside the page on a PDF not printed by Chromium

```bash
cd $W && conda run -n <env> env PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 \
  python $E/anchor_check.py --work <dir> --out $E/anchors_non_chromium.json docs/archive/The_Q_algorithm_a_search_strategy_for_a.pdf
```

- **The file:** `docs/archive/The_Q_algorithm_a_search_strategy_for_a.pdf`, tracked, 19 pages of 490×742
  pt, produced by GPL Ghostscript 9.10, fonts Type 1 and Type 1C (`pdffonts`). It is a third-party paper
  kept in the repository, measured here as a contrast, not proposed for the sheet.
- **Result:** 0 of 1 035 anchors outside the page with pypdf, and 0 of 1 167 with PyMuPDF.
- **Against §4.5:** on the two Chromium-printed files (Type 3 fonts), pypdf put 650 of 874 and 13 068 of
  14 701 anchors outside the page.
- **Not diagnosed:** three files do not establish a cause. The fix went to a RAGIX pull request (§10).

## 8. M2a: the same code and the same bytes on two architectures

**Setup.**

- The ARM64 workstation (aarch64): `lscpu` lists two core types, Cortex-X925 and Cortex-A725, 20 CPUs in
  all.
- A fresh bench folder there, `<bench>/`:
  - the whole tree at 20fdf7a, written by `git archive 20fdf7a`;
  - the evidence scripts;
  - the input bytes staged on x86, each checked by sha256 against `inputs_manifest_m2.json` on arrival: 8
    checked, 0 mismatches.
- The ARM64 workstation's existing environment, used read-only, nothing installed into it. It matches x86 version for
  version: Python 3.12.13, pypdf 6.17.0, PyMuPDF 1.28.0, pypdfium2 5.12.1 with pdfium 152.0.7947.0,
  python-docx 1.2.0, openpyxl 3.1.5, python-pptx 1.0.2, numpy 2.5.1, Pillow 12.3.0, SQLite 3.53.3.
- Both hosts ran `bench_saqqara.py` at sha256 `8f73c50e…`. The code digest over the 319 `ragix_kernels`
  files is equal on both, `ec3c65f45482030d…`.

```bash
# on the ARM64 workstation, in <bench>
env PYTHONPATH=$PWD/code PYTHONDONTWRITEBYTECODE=1 <env>/bin/python scripts/bench_saqqara.py \
  --out results_m2_aarch64.json --work work_m2
# on x86, in a bench holding the same inputs/
conda run -n <env> env PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 python $E/bench_saqqara.py \
  --out $E/results_m2_x86_64.json --work work_m2
cd $E && python3 summarize.py results_m2_x86_64.json results_m2_aarch64.json
```

Run times: the ARM64 workstation from 21:57:17 to 21:58:41, x86 from 21:55:46 to 22:00:00 (+01:00).

**Roots, per file and reader, and for the whole set** (pasted from `summarize.py`):

| input | reader | merkle_root x86_64 | merkle_root aarch64 | equal | path-free tree equal | source_root equal |
|---|---|---|---|---|---|---|
| techsheet_pdf | pymupdf | 9b7a7261c4202c90… | 9b7a7261c4202c90… | yes | yes | yes |
| techsheet_pdf | pypdf | 33006994a33f03f9… | 33006994a33f03f9… | yes | yes | yes |
| slides_pdf | pymupdf | c9f190c800ae25e8… | c9f190c800ae25e8… | yes | yes | yes |
| slides_pdf | pypdf | 633156b3c6589d63… | 633156b3c6589d63… | yes | yes | yes |
| sovereign_pdf | pymupdf | 6f8aaf8acbdaca7d… | 6f8aaf8acbdaca7d… | yes | yes | yes |
| sovereign_pdf | pypdf | cad1473c6aea37f7… | cad1473c6aea37f7… | yes | yes | yes |
| saqqara_readme_md | default | 9b7acc667ab770e5… | 9b7acc667ab770e5… | yes | yes | yes |
| koas_saqqara_md | default | a71aed7b1c518fec… | a71aed7b1c518fec… | yes | yes | yes |
| twin_grid_docx | default | d1e8b9260d70c525… | d1e8b9260d70c525… | yes | yes | yes |
| twin_grid_xlsx | default | 77809574d33c07fd… | 77809574d33c07fd… | yes | yes | yes |
| twin_grid_pptx | default | 3ed22a9b211442ea… | 3ed22a9b211442ea… | yes | yes | yes |
| whole set | pymupdf | 63c2d5b7fd7ebadd… | 63c2d5b7fd7ebadd… | yes | – | yes |
| whole set | pypdf | e7c14b4fc29bfe08… | e7c14b4fc29bfe08… | yes | – | yes |

Code digest equal: True (ec3c65f45482030d… / ec3c65f45482030d…).

**Everything else is identical too.** All fields of the two results files were compared, with wall times,
the renderer label and the host record set aside: two differences, both in the code's git record. The
ARM64 copy came from `git archive` and has no `.git`. Every node count, table, abstention, refusal, drawn
region, caption binding, size-promoted heading, Explorer table and cell, index count and search hit (rank
included) is the same on both hosts.

**Why the inputs were staged again.** The M1 bench lived in a scratch directory that was cleared before
M2a. Staged again:

- the five public files came back byte for byte;
- the three synthetic fixtures did not. Generated twice, two seconds apart: docx and pptx differ only in the
  timestamps of their ZIP entries, and xlsx also in `docProps/core.xml`; the content of every other member is
  identical.

So M2a's synthetic files have new bytes and a new `source_root`, yet the **same `merkle_root` and the same
trees as in M1**. That is the difference between the two roots: `source_root` follows the bytes,
`merkle_root` the content read from them.

**Times**, seconds per page (per file where a file has no pages), run 1 / run 2, pasted from
`summarize.py --timings results_m1_x86_64.json results_m2_x86_64.json results_m2_aarch64.json`:

| input | reader | x86_64, load 2.34→3.74 | x86_64, load 18.40→17.04 | aarch64, load 0.08→0.85 |
|---|---|---|---|---|
| techsheet_pdf | pymupdf | 0.060 / 0.058 | 0.288 / 0.106 | 0.053 / 0.050 |
| techsheet_pdf | pypdf | 0.102 / 0.100 | 0.412 / 0.421 | 0.076 / 0.075 |
| slides_pdf | pymupdf | 0.001 / 0.001 | 0.002 / 0.002 | 0.001 / 0.001 |
| slides_pdf | pypdf | 0.001 / 0.001 | 0.001 / 0.001 | 0.001 / 0.001 |
| sovereign_pdf | pymupdf | 0.118 / 0.104 | 0.185 / 0.174 | 0.081 / 0.080 |
| sovereign_pdf | pypdf | 0.151 / 0.173 | 0.188 / 0.374 | 0.109 / 0.116 |
| saqqara_readme_md | default | 0.002 / 0.002 (per file) | 0.003 / 0.003 (per file) | 0.001 / 0.001 (per file) |
| koas_saqqara_md | default | 0.006 / 0.006 (per file) | 0.011 / 0.011 (per file) | 0.005 / 0.004 (per file) |
| twin_grid_docx | default | 0.033 / 0.007 (per file) | 0.063 / 0.013 (per file) | 0.027 / 0.005 (per file) |
| twin_grid_xlsx | default | 0.048 / 0.004 (per file) | 0.085 / 0.006 (per file) | 0.032 / 0.003 (per file) |
| twin_grid_pptx | default | 0.034 / 0.004 | 0.063 / 0.009 | 0.025 / 0.003 |
| whole set, s | pymupdf / pypdf | 5.168 / 5.094; 7.601 / 8.080 | 30.999 / 9.978; 12.513 / 24.366 | 4.309 / 4.304; 5.886 / 6.324 |

The x86 twin ran while other work held this machine at a one-minute load near 18; its times are recorded,
not interpreted. With the machine lightly loaded (M1, load 2–4, and the ARM64 workstation, load under 1), the ARM64 workstation was
somewhat faster per page on the two text PDFs (the slides PDF takes 0.001 s per page on both).

## 9. M2b: the pinned environment on the ARM64 workstation, and the sheet's figures

**What was installed.**

- **Where:** a Python venv, `<bench>/saqqara-bench`, made from the ARM64 workstation's Python 3.12.13
  (`python -m venv`, with the pip 25.0.1 bundled in it, not upgraded).
- **What:** the 20 packages of `m2b_requirements_pinned.txt` (`pip install --no-deps --require-hashes
  --no-cache-dir`):
  - every dependency of `ragix[saqqara,explorer]` at 20fdf7a;
  - each at the version of the measurement environment, except `pymupdf==1.27.2.2`, the Explorer lock's
    version;
  - each pinned by sha256.
- **Downloads, all from PyPI, 2026-11-16 22:04:00 to 22:04:07 (+01:00):**
  - 19 wheels, 57 212 746 bytes;
  - the odfpy 1.4.1 source archive, 717 045 bytes; it has no wheel and was built locally into
    `odfpy-1.4.1-py2.py3-none-any.whl`;
  - 57 929 791 bytes in all.
  - Building odfpy also fetched its build dependencies into a temporary environment. pip did not record
    which version at this verbosity; it is setuptools by odfpy's build declaration, and the version is not
    measured here.
- `pip check`: "No broken requirements found". The environment takes 203 072 566 bytes on disk (`du -sb`).

| package | version | file | bytes |
|---|---|---|---|
| pymupdf | 1.27.2.2 | pymupdf-1.27.2.2-cp310-abi3-manylinux_2_28_aarch64.whl | 24 318 817 |
| numpy | 2.5.1 | numpy-2.5.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl | 15 167 977 |
| pillow | 12.3.0 | pillow-12.3.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl | 6 266 838 |
| lxml | 6.1.1 | lxml-6.1.1-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl | 4 930 746 |
| pypdfium2 | 5.12.1 | pypdfium2-5.12.1-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl | 3 649 734 |
| pyyaml | 6.0.3 | pyyaml-6.0.3-cp312-cp312-manylinux2014_aarch64…whl | 775 116 |
| odfpy | 1.4.1 | odfpy-1.4.1.tar.gz (source) | 717 045 |
| python-pptx | 1.0.2 | python_pptx-1.0.2-py3-none-any.whl | 472 788 |
| pypdf | 6.17.0 | pypdf-6.17.0-py3-none-any.whl | 388 051 |
| python-docx | 1.2.0 | python_docx-1.2.0-py3-none-any.whl | 252 987 |
| openpyxl | 3.1.5 | openpyxl-3.1.5-py2.py3-none-any.whl | 250 910 |
| xlsxwriter | 3.2.9 | xlsxwriter-3.2.9-py3-none-any.whl | 175 315 |
| certifi | 2026.7.22 | certifi-2026.7.22-py3-none-any.whl | 136 983 |
| urllib3 | 2.7.0 | urllib3-2.7.0-py3-none-any.whl | 131 087 |
| requests | 2.34.2 | requests-2.34.2-py3-none-any.whl | 73 075 |
| idna | 3.19 | idna-3.19-py3-none-any.whl | 68 550 |
| charset-normalizer | 3.4.9 | charset_normalizer-3.4.9-py3-none-any.whl | 64 538 |
| typing-extensions | 4.16.0 | typing_extensions-4.16.0-py3-none-any.whl | 45 571 |
| defusedxml | 0.7.1 | defusedxml-0.7.1-py2.py3-none-any.whl | 25 604 |
| et-xmlfile | 2.0.0 | et_xmlfile-2.0.0-py3-none-any.whl | 18 059 |

Sizes are those of PyPI's file listing (`m2b_planned_downloads_pinned.json`). The hashes pip checked are
in `m2b_install_report.json`.

**The run.**

```bash
# on the ARM64 workstation, in <bench>
env PYTHONPATH=$PWD/code PYTHONDONTWRITEBYTECODE=1 BENCH_ENV_LABEL="saqqara-bench (pinned venv; pymupdf 1.27.2.2)" \
  saqqara-bench/bin/python scripts/bench_saqqara.py --out results_m2b_aarch64_pinned.json --work work_m2b
python3 compare_results.py results_m2_aarch64.json results_m2b_aarch64_pinned.json
python3 scripts/tree_diff.py work_m2/kernel/<id>/pymupdf/run0/stage1/saqqara.json work_m2b/kernel/<id>/pymupdf/run0/stage1/saqqara.json
```

It ran from 22:04:22 to 22:05:51 (+01:00), with `bench_saqqara.py` at sha256 `9b21884a…`. Another job
started on the ARM64 workstation at the same moment (one-minute load 19.81 at the start, 15.33 at the end), so the times
of this run are recorded but not interpreted. The times of §8 stand.

**What changed between PyMuPDF 1.28.0 (M2a) and 1.27.2.2 (M2b), on the same host and the same bytes.**
`compare_results.py` finds 42 differing fields, and all of them lie in the runs of the PyMuPDF reader and
in their labels:

- the recorded reader version, `0.8.0+pymupdf-<version>`, which every tree read with that reader carries in
  its provenance. Hence a different `merkle_root` for the three PDFs, the whole set, and the outline runs;
- the Explorer's extractor label;
- two work-directory paths.

Everything read with the default reader is identical: roots, trees, counts, drawn regions, caption
binding, size headings, index and search.

`tree_diff.py` on the PyMuPDF trees, the version string normalised (`m2b_tree_diff_pymupdf.txt`):

- the tech sheet and the slides: identical trees;
- the operations document: **10 of 14 214 text anchors moved, nothing else.** All ten start a run in a
  Type 3 font at 9.49 pt (an em dash, "with", "(explicit)" …). Under 1.27.2.2 each starts 2.08–2.09 pt
  further left and is 2.08–2.10 pt wider; y moves by −0.08 or +0.67 pt.

**The Explorer's PDF tables at the lock's version:** 10 tables and 59 cells on the tech sheet, 34 tables and
796 cells on the operations document, 0 cells without geometry. These are the same counts as with 1.28.0,
so the §3 figures hold at the version RAGIX declares.

**Which figures the sheet uses.** Counts, roots and tables come from `results_m2b_aarch64_pinned.json`.
Across architectures (§8) and across these two PyMuPDF versions, every count is the same; only the
PyMuPDF-reader roots carry the version. Times come from §8 (M1 x86 and M2a aarch64, both lightly loaded).

## 10. The default reader's text positions, before and after the fix

**What was run.**

- **Before:** `20fdf7a`. **After:** `d859c87`, a local commit on `fix/pdf-text-page-position` (PDF adapter
  0.8.0 → 0.9.0, K2.26).
- **Code:** each commit copied by `git archive`; the fix worktree was only read.
- **Inputs:** M2a's input bytes, plus one contrast file (`extend_inputs.py`; `inputs_manifest_1b.json`):
  `docs/archive/The_Q_algorithm_a_search_strategy_for_a.pdf`, produced by Ghostscript. It is not for the
  sheet.
- **Six runs**, all at `bench_saqqara.py` `576b3605…` with `BENCH_CODE_COMMIT` naming the archived commit:
  - x86 in the measurement environment (load average 24–37, set by other work);
  - the ARM64 workstation in its measurement environment;
  - the ARM64 workstation in `saqqara-bench`.
- **Times are not interpreted**, on either host.
- **A pypdf warning:** in every run, pypdf warned that fontTools is not installed for the Ghostscript PDF's
  Type 1C fonts ("Consider installing fontTools if you encounter encoding problems"). It applies before and
  after alike.

**Architecture parity holds on both commits.** `compare_results.py`, x86 against the ARM64 workstation in the same
environment:

- before: 2 differences, after: 2 differences, both the names of the work directories;
- `before_after.py` gives the same table on both hosts and in both ARM64 environments.

**Per PDF, before → after** (pinned environment; pasted from `1b_before_after_pinned.md`, the rows that
move):

| PDF | measure | before | after |
|---|---|---|---|
| tech sheet | pypdf: text anchors outside the page | 650 / 874 | 0 / 874 |
| tech sheet | headings promoted by size (library pass, default reader) | 73 | 66 |
| operations | pypdf: text anchors outside the page | 13 068 / 14 701 | 76 / 14 701 |
| operations | headings by size: abstained "declared-outline", `would_have_promoted` | 0 | 77 |

The size pass also regroups lines once positions are right:

- tech sheet: 510 → 493 lines;
- operations: 1 738 → 1 785 lines.

**What does not move,** on all four PDFs, both readers:

- sections accepted and rejected;
- headings added by the outline pass;
- abstentions;
- paragraph counts;
- pictures;
- **caption binding** (tech sheet still 1 of 4 bound, 3 abstained `no-candidate-within-gap`);
- drawn ink, regions, refusals and page-furniture exclusions;
- the Explorer's tables;
- the index totals;
- the top five hits of both searches.

**The 76 anchors still outside the page** on the operations document are two per page on all 38 pages:
the running header and the "n / 38" footer. That is the known hole H2: text inside form XObjects.

**The trees, versions ignored** (`tree_diff.py --ignore-versions`, `1b_tree_diff_pinned.txt`):

- tech sheet, pypdf: 874 of 874 anchors moved, and nothing else in the tree;
- operations, pypdf: 14 663 of 14 701 anchors moved, and nothing else;
- the slides (no text layer) and the Ghostscript PDF: identical trees with both readers;
- every PyMuPDF-reader tree: identical.

**Roots.** The `merkle_root` of every PDF changes with both readers, because the reader version (0.8.0 →
0.9.0, with or without `+pymupdf-…`) is recorded in every node's provenance. The Markdown and Office
roots do not change. Both whole-set roots change.

### 10.1 The review commit 2ddf2af, "after" side only

**The commit.**

- `2ddf2af` on top of `d859c87`, same branch, local.
- Its change: a Tj/TJ that shows no character no longer starts a run.
- Code: `ragix_kernels/saqqara/adapters/pdf.py` (the adapter is still 0.9.0), plus the fixture generator, the
  tests, the SPEC and the CHANGELOG.

**The runs.** Each was copied by `git archive` and run on the same inputs, with `BENCH_CODE_COMMIT=2ddf2af`:

- x86, measurement environment: 22:48:47 to 22:50:49;
- the ARM64 workstation, measurement environment: 22:48:47 to 22:50:26;
- the ARM64 workstation, `saqqara-bench`: 22:50:26 to 22:52:04.

**The result: identical to `d859c87`, on every host and in every environment.**

- `compare_results.py`, `d859c87` against `2ddf2af`:
  - 4 differing fields in each of the three pairs: the two work-directory names, the code digest
    (`662068ad…` → `eea8b036…`) and the archived commit;
  - no root, count, table, abstention, region, heading, index or search figure differs.
- `tree_diff.py` on all four PDFs, both readers, both ARM64 environments: every tree equal (16 of 16).
- Architecture parity on `2ddf2af`, x86 against the ARM64 workstation in the measurement environment: 2 differing
  fields, the work-directory names.

### 10.2 The pushed head 1237905, "after" side only

**The commit.**

- `1237905` on top of `2ddf2af`, pushed.
- Its change: a run begun by `'` or `"` is now captured after the operator, which closes H3.
- Code: `adapters/pdf.py` (the adapter is still 0.9.0), plus the fixture generator, a test, the SPEC and the
  CHANGELOG.

**The runs.** The same procedure as §10.1:

- x86, measurement environment: 22:55:51 to 22:57:38;
- the ARM64 workstation, measurement environment: 22:55:41 to 22:57:26;
- the ARM64 workstation, `saqqara-bench`: 22:57:26 to 22:59:05.

**The result: identical to `2ddf2af`, on every host and in every environment.**

- `compare_results.py`, `2ddf2af` against `1237905`:
  - 4 differing fields in each of the three pairs: the two work-directory names, the code digest
    (`eea8b036…` → `218f7dcc…`) and the archived commit;
  - no measured figure differs.
- `tree_diff.py`: 16 of 16 trees equal.
- Architecture parity on `1237905`, x86 against the ARM64 workstation in the measurement environment: 2 differing
  fields, the work-directory names.

None of the public PDFs, nor the contrast PDF, has a run that `1237905` places differently from
`2ddf2af`. Every figure of §10 therefore describes `1237905`, the head the sheet will describe if the pull
request is merged.

**The PR head `8bc11bf`** (checked 2026-11-16 23:01 +01:00, not re-run): `git diff --stat 1237905 8bc11bf`
lists one file, `tests/saqqara/generators.py` (6 insertions, 5 deletions), and `git diff 1237905 8bc11bf --
ragix_kernels` is empty. The reader code is identical, so every figure of §10 also describes `8bc11bf`.

### 10.3 The merge: b7eb0a7 on main

Pull request #42 was merged as `b7eb0a7cc5690d5fce49d1fd7e9a9695a4212df2` (parents `20fdf7a` and `8bc11bf`),
2026-11-17 06:14:34Z (`gh api repos/ovitrac/RAGIX/pulls/42`, read 07:16 +01:00). The merge commit's tree, read
from GitHub, is `e1ec50a74e3f6007e787367356564a778714b042`. That is the tree of `8bc11bf` in the local checkout
(`git rev-parse 8bc11bf^{tree}`). Nothing was fetched into the local repository.

The tree of `b7eb0a7` is the tree of `8bc11bf`, whose reader code is that of `1237905` (§10.2). Every figure of
§10 therefore describes `b7eb0a7`, and so does the Saqqara side of §11. No run was repeated.

## 11. M4: Docling 2.150.1 on the ARM64 workstation, beside Saqqara

**Decisions.**

- The Docling environment runs on the ARM64 workstation. The itemised download list
  (`m4_planned_downloads.json`) was approved, in its minimal, OCR-free form, before anything was fetched.
- The PDF pipeline then failed to import without OpenCV, which the minimal install does not bring. That
  wheel was approved on its own before it was fetched.

**Installed.**

- **Where:** a venv, `<bench>/docling-bench`, made from the ARM64 workstation's Python 3.12.13 (pip
  25.0.1, not upgraded). Nothing changed in the other two environments.
- **Three steps:**
  1. `torch==2.15.0+cpu` and `torchvision==0.29.1+cpu` from the PyTorch CPU index, by sha256
     (`m4_requirements_torch_cpu.txt`, `--no-deps`);
  2. `docling-slim[format-pdf,format-markdown,convert-core,models-local]==2.150.1` from PyPI, with the CPU
     torch wheel's own dependencies named. No CUDA, OCR or second torch package was fetched;
  3. `opencv-python-headless==4.14.0.94`, by sha256 (`m4_requirements_opencv.txt`). docling-ibm-models
     declares it only as an optional extra, `>=4.6.0.66,<5`.
- **Result:** `pip check`: "No broken requirements found". `m4_pip_freeze.txt` lists 73 packages: docling-slim
  2.150.1, docling-core 2.107.1, docling-parse 7.26.0, docling-ibm-models 4.0.4, torch 2.15.0+cpu,
  transformers 5.23.0, huggingface_hub 1.33.0, …
- **What pip fetched** (`m4_downloads.json`): 73 files, 342 210 331 bytes in all.
  - 172 808 949 bytes from download-r2.pytorch.org;
  - 169 401 382 bytes from PyPI, OpenCV's 38 104 309 included.
- **Disk:** the environment takes 1 478 195 047 bytes.

**Models.**

- Fetched by `docling.utils.model_downloader.download_models` with only `with_layout` and
  `with_tableformer` on (formula, picture classifier and RapidOCR off), into the bench folder, on
  2026-11-16 from 23:23:55 to 23:24:10 (+01:00):
  - `docling-project/docling-layout-heron` (main): 171 764 371 bytes;
  - `docling-project/docling-models` (v2.3.0, TableFormer accurate and fast): 358 236 323 bytes;
  - **`docling-project/docling-layout-heron-onnx`: 171 223 607 bytes, which was not in the approved
    plan.**
- Why the ONNX repo came: the layout model's specification names it as an override for the ONNX engine,
  and `with_layout` fetches it. It is unused, because the pipeline's layout engine is torch and onnxruntime
  is not installed.
- Model files: 701 224 301 bytes (`m4_models_files.txt`). M4 downloaded **1 043 434 632 bytes** in all.
- During the runs the Hugging Face hub was set offline, so nothing more could be fetched.

**The run.** `docling_bench.py`, 2026-11-17 07:01:42 to 07:12:58 (+01:00).

- **Pipeline:** Docling's standard PDF pipeline, `do_ocr=False`, table structure on (TableFormer, its default
  `accurate` mode), CPU, Markdown inputs through Docling's Markdown backend.
- **Threads:** `num_threads`, `OMP_NUM_THREADS` and torch's thread count all set to 4, then to 20.
- **Repetition:** one process per file, two runs per thread count.
- **Inputs:** the bytes of `inputs_manifest_1b.json` (the contrast PDF only as a contrast).
- **The machine was not quiet:** other jobs held the one-minute load between 4.08 and 8.58 throughout. The
  times below are comparable with each other only under that load.

**Saqqara's side.**

- Counts come from `results_1b_after3_aarch64_pinned.json`: the pinned environment, reader code of `1237905`,
  the same as `b7eb0a7` on main (§10.3).
- Peak memory comes from `saqqara_rss.py`: the same code and environment, one kernel run per file in its
  own process, 07:13:40 to 07:13:54, after Docling's run.

**Per input** (pasted from `m4_side_by_side.md`):

#### techsheet_pdf

| measure | Saqqara | Docling 2.150.1 |
|---|---|---|
| pages | 19 | 19 |
| elements by type | page 19, paragraph 874 | caption 1, document_index 1, list_item 13, picture 37, section_header 80, table 7, text 108 |
| tables | 10 tables, 59 cells (Explorer, PyMuPDF); rows per table 5, 6, 4, 4, 4, 3, 1, 4, 8, 5 | 7 tables, 125 cells; rows×cols 7×3, 8×3, 6×2, 6×4, 4×2, 10×3, 7×2 |
| headings | 0 declared; +35 by the opt-in outline pass; 66 promoted by size (library) | section_header 80 |
| figures | 4 pictures; 58 drawn regions (library passes) | 37 pictures |
| provenance | every node: source path, format, locator chain (page; for text the anchor x, y, width and font size; for pictures and regions a box), reader and version | 247 of 247 items with prov; bbox 247, charspan 247; origin {'CoordOrigin.BOTTOMLEFT': 247} |
| uncertainty | abstentions: 0; read nodes carry confidence 1.0, inferred ones less | page grades: mean excellent, low good |
| seconds per page, run 1 / run 2 | 0.081 / 0.080 (one process) | 0.522 / 0.534 at 4 threads; 1.839 / 1.766 at 20 threads (conversion only; model load 1.603 / 1.656 s) |
| same output twice | roots equal across runs | JSON, 4 threads, run 1 vs 2: identical; 20 threads: DIFFER; across thread counts: DIFFER (no time stamp in the export) |
| peak resident memory | 79944 KiB | 2343116 KiB (largest over runs) |

#### slides_pdf

| measure | Saqqara | Docling 2.150.1 |
|---|---|---|
| pages | 14 | 14 |
| elements by type | page 14 | picture 27, table 4 |
| tables | 0 tables, 0 cells (Explorer, PyMuPDF); rows per table – | 4 tables, 0 cells; rows×cols 0×0, 0×0, 0×0, 0×0 |
| headings | 0 declared; +0 by the opt-in outline pass; 0 promoted by size (library) | 0 |
| figures | 14 pictures; 0 drawn regions (library passes) | 27 pictures |
| provenance | every node: source path, format, locator chain (page; for text the anchor x, y, width and font size; for pictures and regions a box), reader and version | 31 of 31 items with prov; bbox 31, charspan 31; origin {'CoordOrigin.BOTTOMLEFT': 31} |
| uncertainty | abstentions: saqqara.text_layer:no-text-layer 14; read nodes carry confidence 1.0, inferred ones less | page grades: mean fair, low fair |
| seconds per page, run 1 / run 2 | 0.001 / 0.001 (one process) | 0.473 / 0.529 at 4 threads; 1.743 / 1.640 at 20 threads (conversion only; model load 1.651 / 1.683 s) |
| same output twice | roots equal across runs | JSON, 4 threads, run 1 vs 2: identical; 20 threads: DIFFER; across thread counts: DIFFER (no time stamp in the export) |
| peak resident memory | 73932 KiB | 2397284 KiB (largest over runs) |

#### sovereign_pdf

| measure | Saqqara | Docling 2.150.1 |
|---|---|---|
| pages | 38 | 38 |
| elements by type | heading 117, page 38, paragraph 14701 | caption 4, checkbox_unselected 7, code 66, list_item 174, picture 124, section_header 162, table 30, text 142 |
| tables | 34 tables, 796 cells (Explorer, PyMuPDF); rows per table 4, 3, 2, 3, 5, 6, 5, 5, 4, 6, 6, 4, 22, 11, 18, 5, 8, 2, 5, 8, 5, 4, 5, 9, 6, 5, 7, 3, 5, 3, 6, 5, 5, 8 | 30 tables, 775 cells; rows×cols 4×2, 5×4, 6×3, 5×3, 5×4, 4×4, 6×5, 6×3, 4×3, 22×5, 11×5, 18×5, 5×3, 8×4, 5×6, 8×3, 5×3, 4×3, 5×3, 7×3, 6×3, 5×6, 7×4, 3×4, 5×4, 3×4, 6×3, 5×5, 5×4, 8×2 |
| headings | 117 declared; +24 by the opt-in outline pass; 0 promoted by size (library) | section_header 162 |
| figures | 0 pictures; 0 drawn regions (library passes) | 124 pictures |
| provenance | every node: source path, format, locator chain (page; for text the anchor x, y, width and font size; for pictures and regions a box), reader and version | 709 of 709 items with prov; bbox 709, charspan 709; origin {'CoordOrigin.BOTTOMLEFT': 709} |
| uncertainty | abstentions: 0; read nodes carry confidence 1.0, inferred ones less | page grades: mean excellent, low good |
| seconds per page, run 1 / run 2 | 0.118 / 0.131 (one process) | 0.633 / 0.688 at 4 threads; 4.009 / 3.780 at 20 threads (conversion only; model load 1.662 / 1.741 s) |
| same output twice | roots equal across runs | JSON, 4 threads, run 1 vs 2: identical; 20 threads: DIFFER; across thread counts: DIFFER (no time stamp in the export) |
| peak resident memory | 172856 KiB | 2674696 KiB (largest over runs) |

#### saqqara_readme_md

| measure | Saqqara | Docling 2.150.1 |
|---|---|---|
| pages | 0 | – |
| elements by type | heading 10, paragraph 26 | code 53, list_item 6, section_header 9, table 1, text 79, title 1 |
| tables | 0 (the Markdown reader types no table) | 1 tables, 30 cells; rows×cols 10×3 |
| headings | 10 declared; +0 by the opt-in outline pass | section_header 9, title 1 |
| figures | – | 0 pictures |
| provenance | every node: source path, format, locator chain (page; for text the anchor x, y, width and font size; for pictures and regions a box), reader and version | 0 of 149 items with prov; bbox 0, charspan 0; origin – |
| uncertainty | abstentions: 0; read nodes carry confidence 1.0, inferred ones less | page grades: mean unspecified, low unspecified |
| seconds per page, run 1 / run 2 | 0.001 / 0.001 per file (one process) | 0.015 / 0.015 per file at 4 threads; 0.015 / 0.015 per file at 20 threads (conversion only; model load 0.003 / 0.004 s) |
| same output twice | roots equal across runs | JSON, 4 threads, run 1 vs 2: identical; 20 threads: identical; across thread counts: identical (no time stamp in the export) |
| peak resident memory | 59668 KiB | 420872 KiB (largest over runs) |

#### koas_saqqara_md

| measure | Saqqara | Docling 2.150.1 |
|---|---|---|
| pages | 0 | – |
| elements by type | heading 41, paragraph 162 | code 795, list_item 55, section_header 40, table 32, text 827, title 1 |
| tables | 0 (the Markdown reader types no table) | 32 tables, 580 cells; rows×cols 3×5, 7×3, 5×3, 5×2, 6×3, 10×2, 6×4, 7×2, 6×2, 6×4, 4×2, 7×3, 4×3, 5×3, 7×3, 5×3, 6×3, 4×3, 9×2, 3×3, 6×2, 3×3, 4×4, 6×3, 5×3, 7×2, 2×3, 3×2, 5×2, 12×2, 26×4, 8×3 |
| headings | 41 declared; +0 by the opt-in outline pass | section_header 40, title 1 |
| figures | – | 0 pictures |
| provenance | every node: source path, format, locator chain (page; for text the anchor x, y, width and font size; for pictures and regions a box), reader and version | 0 of 1750 items with prov; bbox 0, charspan 0; origin – |
| uncertainty | abstentions: 0; read nodes carry confidence 1.0, inferred ones less | page grades: mean unspecified, low unspecified |
| seconds per page, run 1 / run 2 | 0.005 / 0.004 per file (one process) | 0.141 / 0.134 per file at 4 threads; 0.140 / 0.134 per file at 20 threads (conversion only; model load 0.004 / 0.004 s) |
| same output twice | roots equal across runs | JSON, 4 threads, run 1 vs 2: identical; 20 threads: identical; across thread counts: identical (no time stamp in the export) |
| peak resident memory | 60712 KiB | 446084 KiB (largest over runs) |

#### qalgo_pdf

| measure | Saqqara | Docling 2.150.1 |
|---|---|---|
| pages | 19 | 19 |
| elements by type | page 19, paragraph 1035 | caption 4, footnote 7, formula 6, list_item 41, picture 1, section_header 19, table 3, text 105 |
| tables | 0 tables, 0 cells (Explorer, PyMuPDF); rows per table – | 3 tables, 212 cells; rows×cols 17×8, 32×3, 2×2 |
| headings | 0 declared; +0 by the opt-in outline pass; 9 promoted by size (library) | section_header 19 |
| figures | 19 pictures; 0 drawn regions (library passes) | 1 pictures |
| provenance | every node: source path, format, locator chain (page; for text the anchor x, y, width and font size; for pictures and regions a box), reader and version | 186 of 186 items with prov; bbox 186, charspan 186; origin {'CoordOrigin.BOTTOMLEFT': 186} |
| uncertainty | abstentions: 0; read nodes carry confidence 1.0, inferred ones less | page grades: mean excellent, low good |
| seconds per page, run 1 / run 2 | 0.021 / 0.021 (one process) | 0.640 / 0.617 at 4 threads; 2.235 / 2.338 at 20 threads (conversion only; model load 1.692 / 1.710 s) |
| same output twice | roots equal across runs | JSON, 4 threads, run 1 vs 2: identical; 20 threads: DIFFER; across thread counts: DIFFER (no time stamp in the export) |
| peak resident memory | 71192 KiB | 2338416 KiB (largest over runs) |

**What these rows do and do not say.**

- **Not the same objects.**
  - A Saqqara PDF paragraph is one text placement as the file draws it. A Docling item is a region its
    layout model found on the rendered page.
  - Saqqara's PDF tables are PyMuPDF's native observations of ruled grids, with a box for every cell.
    Docling's are TableFormer's predictions inside layout-detected table regions.
  - None of these tables has been checked against the pages by eye, so neither count is shown to be right
    here.
- **Pages without text (the slides, OCR off).**
  - Saqqara records one `no-text-layer` abstention per page, with its signals, and reads the 14 pictures.
  - Docling reports 27 pictures and 4 tables with 0 rows, 0 columns and 0 cells.
  - Docling's page grades for that file are "fair". Neither tool reads text there without OCR.
- **Markdown.** Docling types the pipe tables (1 in the README, 32 in the user reference) and the code
  blocks. Saqqara's Markdown reader types headings and paragraphs only (§4.8). Docling's Markdown items
  carry no provenance (0 of 149, 0 of 1 750).
- **Provenance.**
  - Every Docling PDF item carries a page, a bounding box (bottom-left origin) and a character span.
  - Every Saqqara node carries its source, format, locator chain, reader and reader version, and its trees
    are hashed into `merkle_root`.
  - Saqqara's PDF text nodes record an anchor, a width and a font size, not a box.
- **Same output twice** (`m4_docling_json_diff.txt`).
  - At 4 threads, Docling's two JSON exports are byte-identical for every input.
  - At 20 threads they differ between the two runs and from the 4-thread export for every PDF. The counted
    measures stay equal in all four runs, and only bounding-box coordinates move, by at most 0.000412 pt:
    - of pictures and tables in every comparison;
    - and in one comparison also of text items: 9 coordinates on the operations document, 20 threads, run 1
      against run 2.
    - Every differing path is listed in `m4_docling_json_diff.txt` (corrected 2026-11-17: an earlier
      version of this paragraph said pictures and tables only).
  - The export contains no time stamp. The Markdown inputs give identical JSON in every run.
  - Saqqara's roots are equal across runs, across hosts (§8) and, apart from the recorded PyMuPDF version,
    across PyMuPDF versions (§9).
- **Thread count.** At 20 threads Docling was slower than at 4: 1.77–1.84 s per page against 0.52–0.53 s on
  the tech sheet, 3.78–4.01 against 0.63–0.69 on the operations document. This was on a machine with two
  core types and other jobs running; the cause is not diagnosed here.
- **Memory** (`ru_maxrss`, KiB):
  - Docling's peak resident memory per PDF process was 2 148 480–2 674 696 KiB (2.05–2.55 GiB);
  - Saqqara's was 71 192–172 856 KiB (69.5–168.8 MiB) with its default reader (`saqqara_rss_pypdf.json`),
    and 113 832–215 844 KiB (111.2–210.8 MiB) with the PyMuPDF reader.
