"""Write the Saqqara deck's slide files from the dossier's own text (one template, 12 slides).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Usage: python3 make_slides.py saqqara-tech-sheet/slides   (overwrites the 12 slide files)
"""
import pathlib, sys

OUT = pathlib.Path(sys.argv[1])
N = 12

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Work+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.hugeicons.com/font/hgi-stroke-rounded.css">
  <link rel="stylesheet" href="../library/slide-common.css">
  <style>
{css}
    :focus-visible {{ outline: 2px solid #0D62FB; outline-offset: 2px; }}
  </style>
</head>
<body>
  <div class="container">
"""

TAIL = """
  <script src="../library/slide-common.js"></script>
</body>
</html>
"""

CSS_COMMON = """    .card-outer { background: #f3f4f6; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-radius: 12px; padding: 6px; display: flex; flex-direction: column; gap: 4px; min-height: 0; }
    .card-header { display: flex; align-items: center; gap: 8px; padding: 6px 6px 2px; }
    .icon-box { width: 28px; height: 28px; border-radius: 8px; border: 1px solid #e5e7eb; background: #ffffff; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
    .icon-box i { font-size: 15px; line-height: 1; }
    .icon-box.blue i { color: #0D62FB; } .icon-box.indigo i { color: #4338CA; } .icon-box.cyan i { color: #3CBEFA; }
    .card-title { font-size: 16px; font-weight: 600; color: #000000; line-height: 1.25; }
    .card-inner { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 14px; display: flex; flex-direction: column; justify-content: center; flex: 1; min-height: 0; overflow: hidden; font-size: 13.5px; color: #374151; line-height: 1.5; }
    .card-inner strong { color: #000000; font-weight: 600; }
    .card-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; }
    .card-list li { font-size: 15px; color: #374151; line-height: 1.45; padding: 10px 0 10px 18px; position: relative; border-bottom: 1px solid #f3f4f6; }
    .card-list li:last-child { border-bottom: none; }
    .card-list li::before { content: ""; position: absolute; left: 0; top: 18px; width: 6px; height: 6px; border-radius: 50%; background: #0D62FB; }
    .card-list.indigo li::before { background: #4338CA; }
    .card-list strong { color: #000000; font-weight: 600; }
    .card-list code, .card-inner code, td code { font-family: ui-monospace, monospace; font-size: 0.92em; color: #111827; }
    .msg-bar { background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 10px; padding: 11px 16px; display: flex; align-items: center; gap: 10px; font-size: 14px; color: #111827; flex-shrink: 0; line-height: 1.4; }
    .msg-bar i { color: #0D62FB; font-size: 18px; flex-shrink: 0; }
    .msg-bar strong { font-weight: 600; }
    .grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; flex: 1; min-height: 0; }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; flex-shrink: 0; }
    .kpi-card { background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 12px; padding: 6px; display: flex; flex-direction: column; gap: 4px; }
    .kpi-content { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 10px; display: flex; flex-direction: column; gap: 6px; text-align: center; justify-content: center; flex: 1; }
    .kpi-value { font-size: 34px; font-weight: 700; color: #000000; line-height: 1; }
    .kpi-desc { font-size: 12px; color: #6b7280; line-height: 1.35; }
    .slide-table { width: 100%; border-collapse: collapse; }
    .slide-table th { padding: 8px 12px; font-weight: 600; font-size: 12px; text-align: left; background: #ffffff; color: #000000; border-top: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; }
    .slide-table td { padding: 10px 12px; border-bottom: 1px solid #e5e7eb; font-weight: 400; color: #374151; font-size: 14.5px; vertical-align: top; text-align: left; line-height: 1.4; }
    .slide-table td:first-child { font-weight: 600; color: #000000; }"""


def frame(title, h1, subtitle, body, footer, page, css=""):
    return (HEAD.format(title=title, css=CSS_COMMON + ("\n" + css if css else "")) +
            f"""    <div class="diagram-outer">
      <div class="diagram-header">
        <h1>{h1}</h1>
        <p class="subtitle">{subtitle}</p>
      </div>

      <div class="slide-body">
{body}
      </div>
    </div>

    <div class="footer-line"></div>
    <div class="slide-footer">
      <div class="footer-message">{footer}</div>
      <span class="footer-page">{page} / {N}</span>
    </div>
  </div>
""" + TAIL)


def card(icon, color, title, inner):
    return f"""          <div class="card-outer">
            <div class="card-header">
              <span class="icon-box {color}"><i class="hgi-stroke {icon}" aria-hidden="true"></i></span>
              <span class="card-title">{title}</span>
            </div>
            <div class="card-inner">
{inner}
            </div>
          </div>"""


def kpis(items):
    cells = "\n".join(f"""          <div class="kpi-card"><div class="kpi-content"><div class="kpi-value">{v}</div><div class="kpi-desc">{d}</div></div></div>"""
                      for v, d in items)
    return f"""        <div class="kpi-grid">
{cells}
        </div>"""


def ul(items, cls=""):
    lis = "\n".join(f"                <li>{x}</li>" for x in items)
    return f"""              <ul class="card-list {cls}">
{lis}
              </ul>"""


def table(head, rows, widths=None):
    cols = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    th = "".join(f"<th>{h}</th>" for h in head)
    trs = "\n".join("              <tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"""            <table class="slide-table">
              <colgroup>{cols}</colgroup>
              <thead><tr>{th}</tr></thead>
              <tbody>
{trs}
              </tbody>
            </table>"""


def msg(icon, html):
    return f"""        <div class="msg-bar"><i class="hgi-stroke {icon}" aria-hidden="true"></i><span>{html}</span></div>"""


slides = {}

# 1 ── cover
slides["slide-01-cover.html"] = HEAD.format(title="Slide 01 — Saqqara cover", css="""    .diagram-outer { padding-bottom: 0; }
    .cover-layout { flex: 1; display: flex; align-items: stretch; gap: 0; }
    .cover-left { width: 44%; display: flex; flex-direction: column; padding: 30px 40px 14px 8px; gap: 14px; }
    .cover-logo { width: 150px; height: auto; }
    .cover-title { font-family: 'Instrument Serif', serif; font-size: 64px; font-weight: 500; color: #000000; letter-spacing: 0.02em; line-height: 1.05; margin-top: auto; }
    .cover-comment { font-size: 15px; font-weight: 400; color: #6b7280; line-height: 1.55; margin-top: 10px; }
    .cover-badges { display: flex; gap: 6px; margin-top: auto; }
    .cover-badge { flex: 1; padding: 5px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; text-align: center; color: #ffffff; }
    .cover-badge.blue { background: linear-gradient(135deg, #0D62FB, #3CBEFA); }
    .cover-badge.indigo { background: linear-gradient(135deg, #4338CA, #6366F1); }
    .cover-badge.cyan { background: linear-gradient(135deg, #3CBEFA, #7DD3FC); }
    .cover-divider { width: 100%; height: 1px; background: #e5e7eb; margin-bottom: 10px; }
    .cover-meta { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .cover-meta-item { font-size: 11px; color: #6b7280; }
    .cover-right { width: 56%; position: relative; padding: 18px; }
    .cover-img-wrapper { position: absolute; top: 18px; right: 18px; bottom: 18px; left: 0; border-radius: 12px; overflow: hidden; border: 1px solid #e5e7eb; background: #f3f4f6; }
    .cover-img-wrapper img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; object-position: center top; display: block; }
    .cover-tagline { position: absolute; bottom: 20px; right: 20px; left: 20px; font-family: 'Instrument Serif', serif; font-size: 19px; font-style: italic; line-height: 1.4; color: rgba(255,255,255,0.95); background: rgba(10,37,64,0.55); border: 1px solid rgba(255,255,255,0.25); padding: 10px 18px; border-radius: 6px; z-index: 1; }""") + """    <div class="diagram-outer">
      <div class="cover-layout">
        <div class="cover-left">
          <img class="cover-logo" src="../images/logo.png" alt="Adservio">

          <div class="cover-title">Saqqara</div>

          <div class="cover-comment">The document substrate of RAGIX: PDF, Word, Excel, PowerPoint and Markdown read into one typed tree, with an exact citation on every node, by tested rules and without any language model.</div>

          <div class="cover-badges">
            <span class="cover-badge blue">Model-free</span>
            <span class="cover-badge indigo">Citeable</span>
            <span class="cover-badge cyan">Replayable</span>
          </div>

          <div class="cover-footer">
            <div class="cover-divider"></div>
            <div class="cover-meta">
              <div class="cover-meta-item">November 2026</div>
              <div class="cover-meta-item">RAGIX 0.74 &middot; commit b7eb0a7</div>
              <div class="cover-meta-item">Adservio Innovation Lab</div>
            </div>
          </div>
        </div>

        <div class="cover-right">
          <div class="cover-img-wrapper">
            <img src="../images/hero.png" alt="The nodes Saqqara records on one page of a public document, drawn as boxes without the page.">
            <div class="cover-tagline">One public page as Saqqara records it: every node, drawn without the page.</div>
          </div>
        </div>
      </div>
    </div>
  </div>
""" + TAIL

# 2 ── contents
items = [("The problem", "Two questions every reader of a citation asks"),
         ("Five differentiators", "Provenance, abstention, no model, one tree, replay"),
         ("From bytes to a citeable tree", "Read, build, recognise, index, explore"),
         ("Inside RAGIX", "Six kernels, four interfaces, the families above"),
         ("Saqqara and Docling", "Two approaches, compared and measured"),
         ("Proof, maturity &amp; openness", "Measured on public documents")]
grid = "\n".join(f"""        <div class="sommaire-item">
          <span class="sommaire-num">{i:02d}</span>
          <div class="sommaire-text">
            <div class="sommaire-section-title">{t}</div>
            <div class="sommaire-section-desc">{d}</div>
          </div>
        </div>""" for i, (t, d) in enumerate(items, 1))
slides["slide-02-contents.html"] = HEAD.format(title="Slide 02 — Contents", css="""    .diagram-outer { display: flex; flex-direction: column; padding: 48px 56px; }
    .sommaire-header { margin-bottom: 36px; }
    .sommaire-header h1 { font-family: 'Instrument Serif', serif; font-size: 36px; font-weight: 400; color: #111827; }
    .sommaire-grid { display: grid; grid-template-rows: repeat(3, auto); grid-auto-flow: column; gap: 20px 48px; flex: 1; align-content: center; }
    .sommaire-item { display: flex; align-items: flex-start; gap: 16px; padding: 16px 20px; background: #ffffff; border-radius: 10px; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .sommaire-num { font-size: 28px; font-weight: 200; color: #d1d5db; line-height: 1; flex-shrink: 0; width: 36px; }
    .sommaire-text { display: flex; flex-direction: column; gap: 4px; }
    .sommaire-section-title { font-size: 17px; font-weight: 600; color: #111827; }
    .sommaire-section-desc { font-size: 12px; color: #6b7280; line-height: 1.4; }""") + f"""    <div class="diagram-outer">
      <div class="sommaire-header">
        <h1>Contents</h1>
      </div>

      <div class="sommaire-grid">
{grid}
      </div>
    </div>
  </div>
""" + TAIL

# 3 ── in one minute
body = kpis([("0", "Language-model calls"), ("5", "Formats read natively"),
             ("6", "RAGIX kernels"), ("154", "Propositions, each tested")]) + """
        <div class="grid-2col">
""" + card("hgi-alert-circle", "blue", "The problem", ul([
    "Most pipelines flatten documents into strips of text: a table loses its rows and columns, a heading its place, a citation shrinks to a file name.",
    "Layout models recover part of that structure by prediction. A prediction can be wrong without saying so.",
    "The same file can read differently after a model or library update.",
])) + "\n" + card("hgi-checkmark-circle-02", "indigo", "Four principles, each checked by the tests", ul([
    "<strong>Object before fragment</strong>: one typed tree per document, the same vocabulary for every format.",
    "<strong>An exact citation on every node</strong>: page and position, sheet and cell, slide and shape, line; plus the reader's version.",
    "<strong>Abstain rather than guess</strong>: undecided cases are recorded with place, reason and signals.",
    "<strong>Reproducible by construction</strong>: canonical JSON, same files, same trees, same root hash.",
], "indigo")) + """
        </div>
""" + msg("hgi-help-circle", "<strong>Where exactly does this statement come from? Do I get the same reading tomorrow, on another machine?</strong> Saqqara records the answer to both with the reading itself.")
slides["slide-03-one-minute.html"] = frame("Slide 03 — Saqqara in one minute", "Saqqara in one minute",
    "The part of RAGIX that reads documents, and keeps the exact place every node came from", body,
    "“Read it once, exactly, keep every citation”", 3)

# 4 ── differentiators
body = """        <div style="flex:1; min-height:0;">
""" + table(["Differentiator", "What it means", "Value"], [
    ["Provenance on every node", "Page and position, cell, paragraph, slide or line, plus the reader and its version", "Every statement can be traced to its exact source"],
    ["Rules that abstain", "Ordered rules over closed vocabularies; undecided cases are recorded, not filled in", "No silent error presented as structure"],
    ["No model inside", "Reading and structure are computed; the only model is an optional embedder", "Runs anywhere, sovereign, nothing to host"],
    ["One tree for every format", "Five native formats, plus legacy Word and Excel through a recorded conversion", "One citation scheme for a mixed corpus"],
    ["Reproducible and accounted", "Canonical JSON, stable root hashes, every observation a node or a counted drop", "A reading can be replayed and audited"],
], ["24%", "46%", "30%"]) + """
        </div>
""" + msg("hgi-document-validation", "Four of the five are propositions of the specification, each carried by a test; the fifth, no model inside, is a design rule that can be read in the dependencies. The specification also publishes its two known gaps, H1 and H2.")
slides["slide-04-differentiators.html"] = frame("Slide 04 — Five differentiators", "What makes Saqqara unique",
    "Five design choices that hold together", body, "“A document read by Saqqara comes with its own audit trail”", 4)

# 5 ── capabilities
steps = [("hgi-file-02", "Read", "PDF, Word, Excel, PowerPoint, Markdown; legacy .doc and .xls through a recorded conversion", "Raw observations with their exact locators"),
         ("hgi-tree-01", "Build", "One typed tree per document", "Every observation as a node, an attachment or a counted drop"),
         ("hgi-grid-table", "Recognise", "Tables, header bands, label columns, value islands, sections", "Inferred structure with its confidence, or an abstention"),
         ("hgi-database", "Index &amp; search", "Chunks in one SQLite store; lexical lane, optional dense lane, rank fusion", "Hits carrying both lane ranks and their nodes"),
         ("hgi-file-search", "Explore", "Profile a document set, read its fields and tables, report", "A replayable reading with privacy marks")]
cols = "\n".join(f"""          <div class="step">
            <div class="step-head"><span class="icon-box blue"><i class="hgi-stroke {i}" aria-hidden="true"></i></span><span class="step-num">{n:02d}</span></div>
            <div class="step-title">{t}</div>
            <div class="step-what">{w}</div>
            <div class="step-rec"><span>Recorded</span>{r}</div>
          </div>""" for n, (i, t, w, r) in enumerate(steps, 1))
body = f"""        <div class="steps">
{cols}
        </div>
""" + msg("hgi-check-list", "<strong>Everything not done is counted:</strong> files refused with their reason, the abstention register, the drops of the builder and of each analyzer, the chunker's and the embedder's refusals. A run's summary is computed from those registers, so it cannot claim more than was done.")
slides["slide-05-capabilities.html"] = frame("Slide 05 — From bytes to a citeable tree", "From bytes to a citeable tree",
    "Five capabilities, performed in order on every document", body,
    "“Each step marks what it read, what it inferred, and what it could not handle”", 5,
    css="""    .steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; flex: 1; min-height: 0; }
    .step { background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; display: flex; flex-direction: column; gap: 10px; min-height: 0; }
    .step-head { display: flex; align-items: center; justify-content: space-between; }
    .step-num { font-size: 26px; font-weight: 200; color: #9ca3af; line-height: 1; }
    .step-title { font-size: 21px; font-weight: 600; color: #000000; }
    .step-what { font-size: 16px; color: #374151; line-height: 1.5; flex: 1; }
    .step-rec { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; font-size: 14.5px; color: #374151; line-height: 1.4; }
    .step-rec span { display: block; font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #0D62FB; margin-bottom: 3px; }""")

# 6 ── inside RAGIX
body = """        <div class="grid-2col">
""" + card("hgi-layers-01", "blue", "Six kernels, one contract", """            <table class="slide-table">
              <thead><tr><th>Kernel</th><th>Requires &rarr; provides</th></tr></thead>
              <tbody>
              <tr><td><code>saqqara</code></td><td>&mdash; &rarr; document_tree, traces, merkle_root</td></tr>
              <tr><td><code>saqqara_index</code></td><td>document_tree &rarr; document_store</td></tr>
              <tr><td><code>explorer_census</code></td><td>&mdash; &rarr; document_census</td></tr>
              <tr><td><code>explorer_profile</code></td><td>census &rarr; document_profiles</td></tr>
              <tr><td><code>explorer_read</code></td><td>profile &rarr; document_readings</td></tr>
              <tr><td><code>explorer_report</code></td><td>read &rarr; document_reading_report</td></tr>
              </tbody>
            </table>""") + "\n" + card("hgi-route-01", "indigo", "Four doors, one answer, and the families above", ul([
    "<strong>MCP</strong>: four tools, <code>koas_saqqara_run</code>, <code>_status</code>, <code>_index</code>, <code>_search</code>.",
    "<strong>Command line</strong>: <code>saqqaractl run | status | index | search</code>, each with <code>--json</code>; the same hits as MCP, checked by a test.",
    "<strong>Python</strong>: <code>SaqqaraKernel().run(...)</code> returns the envelope of every RAGIX kernel.",
    "<strong>The orchestrator</strong>: <code>ragix-koas</code> chains Saqqara with the other families.",
    "<strong>Built on it</strong>: the tender family probes the store Saqqara builds; the harvest library reports over its readings.",
], "indigo")) + """
        </div>
""" + msg("hgi-git-merge", "<strong>Chained by what they declare.</strong> The registry resolves a requirement by kernel name, then by a capability exactly one kernel provides; two providers is an error, never a silent choice. Every run records which kernel served it.")
slides["slide-06-ragix.html"] = frame("Slide 06 — Saqqara inside RAGIX", "Saqqara inside RAGIX",
    "Not a separate product: a family of RAGIX kernels, and the substrate the document families stand on", body,
    "“The reading is shared; the interpretation stays with each consumer”", 6)

# 7 ── Docling, two approaches
body = """        <div style="flex:1; min-height:0;">
""" + table(["Point", "Saqqara", "Docling 2.150.1"], [
    ["Method", "Reads the file's own records; recognises structure by ordered rules", "Predicts layout with a model derived from RT-DETR; table structure with TableFormer"],
    ["Models in the reading", "None; an embedding model only for the optional dense search lane", "Layout and table models by default; optional OCR, vision-language and speech models"],
    ["Formats", "PDF, Word, Excel, PowerPoint, Markdown; legacy .doc and .xls", "PDF, Office, HTML, EPUB, images, audio, LaTeX, e-mail and more"],
    ["Scanned pages", "Not read: recorded as unread", "Read with OCR (several engines)"],
    ["Citation", "A locator chain on every node, plus the reader's version", "Page, bounding box and character span on PDF items"],
    ["Uncertainty", "Read or inferred on each node; an abstention register with place and reason", "Confidence grades per page and per document"],
    ["Licences", "MIT; default dependencies MIT, BSD and Apache", "MIT code; models under their own licences"],
], ["18%", "41%", "41%"]) + """
        </div>
""" + msg("hgi-justice-scale-01", "<strong>Docling predicts</strong> the layout of each page with trained vision models; <strong>Saqqara reads</strong> what the file records and recognises structure with written rules. Both run locally; both are MIT-licensed.")
slides["slide-07-docling.html"] = frame("Slide 07 — Saqqara and Docling", "Saqqara and Docling",
    "Two answers to one question, from opposite directions; Docling's column from its public documentation", body,
    "“Docling sees a page the way a person does; Saqqara states what a file records”", 7)

# 8 ── measured side by side
pairs = [("Seconds per page", "0.08 / 0.12&ndash;0.13", "0.52&ndash;0.53 / 0.63&ndash;0.69", "reading; Saqqara's Explorer adds 0.06 / 0.07 for PDF tables"),
         ("Peak memory per process", "78.1 / 168.8 MiB", "2.15 / 2.46 GiB", "Docling also loads its models: 1.6&ndash;1.7 s per process"),
         ("Tables &middot; cells", "10 &middot; 59 / 34 &middot; 796", "7 &middot; 125 / 30 &middot; 775", "grids drawn in the file vs. TableFormer's predictions"),
         ("Same output twice", "Roots equal", "JSON byte-identical", "at 4 threads; at 20, Docling's boxes moved by at most 0.000412 pt")]
cards = "\n".join(f"""          <div class="cmp">
            <div class="cmp-label">{l}</div>
            <div class="cmp-row"><span class="who s">Saqqara</span><span class="val">{a}</span></div>
            <div class="cmp-row"><span class="who d">Docling</span><span class="val">{b}</span></div>
            <div class="cmp-note">{n}</div>
          </div>""" for l, a, b, n in pairs)
body = f"""        <div class="cmp-grid">
{cards}
        </div>
""" + msg("hgi-information-circle", "Two public PDFs from the RAGIX repository (19 and 38 pages), one ARM64 machine, CPU only, two runs each; Docling with OCR off and 4 threads. <strong>The two tools do not produce the same objects, and no table has been checked against the pages by eye:</strong> a higher count is not shown to be a better reading.")
slides["slide-08-measured.html"] = frame("Slide 08 — Measured side by side", "Measured side by side",
    "The same public files, the same machine &mdash; values for the 19-page / 38-page PDF", body,
    "“Counts are observations, not a score”", 8,
    css="""    .cmp-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; flex: 1; min-height: 0; }
    .cmp { background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; display: flex; flex-direction: column; justify-content: center; gap: 14px; min-height: 0; }
    .cmp-label { font-size: 17px; font-weight: 600; color: #000000; }
    .cmp-row { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px 14px; display: flex; flex-direction: column; gap: 4px; }
    .who { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
    .who.s { color: #0D62FB; } .who.d { color: #4338CA; }
    .val { font-size: 24px; font-weight: 700; color: #000000; line-height: 1.2; }
    .cmp-note { font-size: 13.5px; color: #4b5563; line-height: 1.45; }""")

# 9 ── when to use which
body = """        <div class="grid-2col">
""" + card("hgi-image-01", "indigo", "Docling is the better fit when&hellip;", ul([
    "pages are scanned or photographed;",
    "the corpus includes HTML, EPUB, images or audio;",
    "reading order must be recovered from complex visual layouts;",
    "the goal is clean Markdown or JSON for a retrieval framework it already plugs into.",
], "indigo")) + "\n" + card("hgi-location-01", "blue", "Saqqara is the better fit when&hellip;", ul([
    "every statement must be cited to its page, cell or line;",
    "uncertain structure must be flagged rather than predicted;",
    "a reading must be replayable and its hash checked later;",
    "spreadsheets matter cell by cell, and no model may be hosted or updated.",
])) + """
        </div>
""" + msg("hgi-share-08", "<strong>Using both.</strong> A page Saqqara records as unread, because it has no text layer, is exactly the page an OCR pipeline such as Docling's can read, provided each reading keeps the citation of the tool that produced it. RAGIX does not wire Docling in today: this is a pattern, not a feature.")
slides["slide-09-which.html"] = frame("Slide 09 — When to use which", "When to use which",
    "Complementary rather than exclusive", body, "“A pattern, not a feature”", 9)

# 10 ── proof
body = kpis([("0", "Measured figures that differ between x86-64 and ARM64"),
             ("1&nbsp;850", "Tests passed at b7eb0a7 (9 skipped, 1 expected failure)"),
             ("14", "Pages without text, recorded as unread &mdash; no text invented"),
             ("650&nbsp;&rarr;&nbsp;0", "Text anchors off the page, fixed while measuring")]) + """
        <div class="grid-2col">
""" + card("hgi-repeat", "blue", "Same bytes, same reading", ul([
    "Each file read twice: the same <code>merkle_root</code> over the trees and the same <code>source_root</code> over the bytes.",
    "x86-64 and ARM64 agreed on every root, count, table, abstention, index count and search hit, rank included.",
    "Five public files from the RAGIX repository: the tech sheet, its slide deck, a 38-page operations guide, two Markdown files.",
])) + "\n" + card("hgi-search-02", "indigo", "Found and fixed while measuring", ul([
    "The default PDF reader placed text without the page transformation: 650 of 874 and 13&nbsp;068 of 14&nbsp;701 anchors off the page.",
    "A new proposition, K2.26, and its test; after the fix (pull request #42): 0 and 76.",
    "The 76 are a running header and footer drawn in form objects: the published known gap H2.",
], "indigo")) + """
        </div>"""
slides["slide-10-proof.html"] = frame("Slide 10 — Proof", "Proof, measured on public documents",
    "Anyone can repeat it: every file is published in the RAGIX repository", body,
    "“Specification, test, fix, re-measure”", 10)

# 11 ── maturity and limits
body = """        <div class="grid-2col">
""" + card("hgi-check-list", "blue", "Maturity by layer", """            <table class="slide-table">
              <tbody>
              <tr><td>Nodes, provenance, canonical JSON</td><td>Gated &middot; 10 propositions</td></tr>
              <tr><td>Readers: PDF, Word, Excel, PowerPoint, Markdown</td><td>Gated &middot; 26</td></tr>
              <tr><td>Builder and structure analyzers</td><td>Gated &middot; 73</td></tr>
              <tr><td>Run envelope, roots and registers</td><td>Gated &middot; 3</td></tr>
              <tr><td>Figures, captions, drawn regions</td><td>Gated as a library &middot; 20</td></tr>
              <tr><td>Store and hybrid search</td><td>Gated &middot; 22</td></tr>
              <tr><td>Field views and the Explorer</td><td>Specified and tested &middot; 29</td></tr>
              <tr><td>Legacy Word and Excel</td><td>Specified and tested &middot; 5</td></tr>
              </tbody>
            </table>""") + "\n" + card("hgi-unavailable", "indigo", "What it does not do yet", ul([
    "PDF text carries a placement point, not a box; one printed line can become several nodes.",
    "A standard run builds no PDF tables (the Explorer reads them) and no figures, captions or drawn regions.",
    "Markdown pipe tables are kept as raw text.",
    "No OCR: scanned pages are reported as unread, never as empty.",
    "Known gaps H1 (undecodable spreadsheet images) and H2 (text in PDF form objects).",
], "indigo")) + """
        </div>
""" + msg("hgi-license", "<strong>Open and sovereign.</strong> MIT, on GitHub with its specification and tests. The default install carries MIT, BSD and Apache licences only; PyMuPDF (AGPL-3.0 or commercial), an embedding model and LibreOffice are optional, by name.")
slides["slide-11-maturity.html"] = frame("Slide 11 — Maturity", "Maturity, limits and openness",
    "Stated layer by layer against the specification, limits included", body,
    "“Limits are listed rather than left to be found”", 11)

# 12 ── closing
body = """        <div class="grid-2col">
""" + card("hgi-star", "blue", "Why Saqqara", ul([
    "<strong>Citeable</strong>: an exact address for every node, down to the cell and the line.",
    "<strong>Honest</strong>: it abstains, refuses and counts, and says where.",
    "<strong>Replayable</strong>: the same files give the same trees and hashes, on any machine.",
    "<strong>Sovereign</strong>: no language model, no service to call, permissive licences by default.",
])) + "\n" + card("hgi-mail-01", "indigo", "Find out more", ul([
    "Inside RAGIX 0.74, open source: github.com/ovitrac/RAGIX",
    "From a RAGIX checkout: <code>pip install -e \".[saqqara]\"</code>",
    "Specification: <code>ragix_kernels/saqqara/SPEC.md</code>",
    "Olivier Vitrac, PhD, HDR &mdash; olivier.vitrac@adservio.fr",
    "Adservio Innovation Lab &mdash; Tour Franklin, 100-101 Terrasse Boieldieu, 92800 Puteaux &mdash; La D&eacute;fense",
], "indigo")) + """
        </div>"""
slides["slide-12-closing.html"] = frame("Slide 12 — Closing", "Read once, cite everywhere",
    "The document layer every RAGIX family stands on", body,
    "“A document is an object before it is a fragment”", 12)

assert len(slides) == N, len(slides)
OUT.mkdir(parents=True, exist_ok=True)
for name, html in slides.items():
    (OUT / name).write_text(html, encoding="utf-8")
print("\n".join(slides))
