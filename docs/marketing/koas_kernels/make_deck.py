#!/usr/bin/env python3
"""Generate the slides of the "Agent-Operated Kernels" deck.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Every slide shares one template (the RAGIX deck framework in kernels-deck/library).
Each slide also carries speaker notes (NOTES), which the single-file player built by
deploy.py shows on demand and uses as the narration of its autoplay mode.

Usage:  python make_deck.py      (writes kernels-deck/slides/*.html and notes.json)
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = os.path.join(HERE, "kernels-deck", "slides")

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Work+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.hugeicons.com/font/hgi-stroke-rounded.css">
  <link rel="stylesheet" href="../library/slide-common.css">
  <style>
    .intro-card {{ background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 12px; padding: 11px 16px; font-size: 14px; color: #374151; line-height: 1.5; flex-shrink: 0; }}
    .intro-card strong {{ color: #000; font-weight: 600; }}
    .row {{ display: flex; gap: 16px; align-items: stretch; min-height: 0; flex: 1; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; flex: 1; min-height: 0; }}
    .card-outer {{ flex: 1; background: #f3f4f6; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-radius: 12px; padding: 6px; display: flex; flex-direction: column; gap: 4px; min-height: 0; }}
    .card-header {{ display: flex; align-items: center; gap: 8px; padding: 6px 6px 2px; }}
    .icon-box {{ width: 28px; height: 28px; border-radius: 8px; border: 1px solid #e5e7eb; background: #fff; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
    .icon-box i {{ font-size: 15px; line-height: 1; color: #0D62FB; }}
    .card-title {{ font-size: 16px; font-weight: 600; color: #000; line-height: 1.25; }}
    .card-inner {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 11px 14px; display: flex; flex-direction: column; gap: 6px; flex: 1; min-height: 0; overflow: hidden; }}
    .card-desc {{ font-size: 13px; color: #4b5563; line-height: 1.5; }}
    .card-desc strong {{ color: #000; font-weight: 600; }}
    ul.pts {{ margin: 0; padding-left: 18px; font-size: 13px; color: #4b5563; line-height: 1.55; }}
    ul.pts li {{ margin: 2px 0; }}
    code, .mono {{ font-family: 'JetBrains Mono', Menlo, monospace; font-size: 12px; color: #1e3a8a; }}
    pre.code {{ margin: 0; font-family: 'JetBrains Mono', Menlo, monospace; font-size: 12.5px; line-height: 1.5; color: #111827; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 14px; white-space: pre; overflow: hidden; }}
    pre.code .k {{ color: #4338CA; font-weight: 600; }}
    pre.code .c {{ color: #6b7280; }}
    table.t {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    table.t th {{ text-align: left; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: #6b7280; font-weight: 600; padding: 7px 10px; border-bottom: 1px solid #e5e7eb; }}
    table.t td {{ padding: 8px 10px; border-bottom: 1px solid #eef0f3; color: #374151; vertical-align: top; line-height: 1.4; }}
    table.t td:first-child {{ color: #000; font-weight: 600; white-space: nowrap; }}
    .flow {{ display: flex; align-items: stretch; gap: 0; flex-shrink: 0; }}
    .step {{ flex: 1; background: #fff; border: 1px solid #dbeafe; border-radius: 10px; padding: 10px 12px; }}
    .step b {{ display: block; font-size: 14px; color: #0D62FB; margin-bottom: 4px; }}
    .step span {{ font-size: 12.5px; color: #4b5563; line-height: 1.45; display: block; }}
    .arrow {{ width: 28px; display: flex; align-items: center; justify-content: center; color: #9ca3af; font-size: 18px; flex-shrink: 0; }}
    .lane {{ display: flex; gap: 10px; align-items: center; }}
    .lane .lbl {{ width: 120px; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #6b7280; font-weight: 600; flex-shrink: 0; }}
    .chip {{ font-family: 'JetBrains Mono', Menlo, monospace; font-size: 12.5px; background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; border-radius: 999px; padding: 5px 12px; }}
    .chip.asked {{ background: #0D62FB; color: #fff; border-color: #0D62FB; }}
    .note {{ font-size: 12.5px; color: #6b7280; line-height: 1.5; }}
    .big {{ font-family: 'Instrument Serif', serif; font-size: 40px; color: #0D62FB; line-height: 1; }}
    .kpis {{ display: flex; gap: 14px; flex-shrink: 0; }}
    .kpi {{ flex: 1; background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px 14px; }}
    .kpi .lab {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
    .cover {{ display: flex; flex-direction: column; justify-content: center; height: 100%; padding: 0 70px; background: radial-gradient(1200px 600px at 85% 20%, #dbeafe 0%, #ffffff 60%); }}
    .cover h1 {{ font-family: 'Instrument Serif', serif; font-size: 76px; font-weight: 400; margin: 0 0 10px; color: #000; }}
    .cover p.sub {{ font-size: 22px; color: #374151; max-width: 900px; line-height: 1.4; margin: 0 0 34px; }}
    .cover p.koas {{ font-size: 15px; color: #4b5563; margin: -18px 0 30px; }}
    .cover p.koas strong {{ color: #0D62FB; font-weight: 600; }}
    .cover .meta {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .cover .meta span {{ font-size: 13px; color: #1e40af; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 999px; padding: 6px 14px; }}
    .cover .by {{ position: absolute; bottom: 40px; left: 70px; font-size: 13px; color: #6b7280; }}
    :focus-visible {{ outline: 2px solid #0D62FB; outline-offset: 2px; }}
  </style>
</head>
<body>
"""

PAGE = """  <div class="container">
    <div class="diagram-outer">
      <div class="diagram-header">
        <h1>{h1}</h1>
        <p class="subtitle">{sub}</p>
      </div>

      <div class="slide-body">
{body}
      </div>
    </div>

    <div class="footer-line"></div>
    <div class="slide-footer">
      <div class="footer-message">"{foot}"</div>
      <span class="footer-page">{n} / {total}</span>
    </div>
  </div>

  <script src="../library/slide-common.js"></script>
</body>
</html>
"""


def card(icon: str, title: str, inner: str) -> str:
    return f"""          <div class="card-outer">
            <div class="card-header"><span class="icon-box"><i class="hgi-stroke {icon}" aria-hidden="true"></i></span><span class="card-title">{title}</span></div>
            <div class="card-inner">{inner}</div>
          </div>
"""


def pts(*items: str) -> str:
    return '<ul class="pts">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def desc(text: str) -> str:
    return f'<div class="card-desc">{text}</div>'


def flow(*steps: tuple[str, str]) -> str:
    parts = []
    for i, (b, s) in enumerate(steps):
        if i:
            parts.append('<div class="arrow">&rarr;</div>')
        parts.append(f'<div class="step"><b>{b}</b><span>{s}</span></div>')
    return '        <div class="flow">' + "".join(parts) + "</div>\n"


def table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'        <table class="t"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>\n'


S: list[dict] = []

# 1 ── cover ─────────────────────────────────────────────────────────────────────────
S.append(dict(file="slide-01-cover.html", cover=True, title="Agent-Operated Kernels",
  notes="Agent-operated kernels: validated programs whose operator is a language model, not a person. "
        "This deck explains the concept as it is built in RAGIX, under the name KOAS, for Kernel-Orchestrated "
        "Audit System. Code audit was the original intent, and the name kept it; the concept itself is not "
        "limited to audit. Every claim points to the code."))

# 2 ── the gap ───────────────────────────────────────────────────────────────────────
S.append(dict(file="slide-02-gap.html", h1="The gap",
  sub="What is missing between a capable model and a trustworthy result",
  foot="Fluent models, atomic tools, hand-written glue",
  body='        <div class="intro-card">A coding-capable model can plan, call tools and write. What it cannot do on its own is <strong>compute exactly, reproduce a result, and prove how it was obtained</strong>, on material larger than its context.</div>\n'
       '        <div class="row">\n'
       + card("hgi-ai-brain-01", "Models are fluent, not exact",
              pts("A number written by a model can drift between runs", "A corpus does not fit in a context window",
                  "Nothing records why an answer is what it is"))
       + card("hgi-wrench-01", "Tools are atomic",
              pts("A function call does one thing", "It declares no dependency on other tools",
                  "It returns a value, not a provenance"))
       + card("hgi-git-branch", "Pipelines are hand-written",
              pts("Every new combination of tools is new glue code", "The model re-plans each step in prose",
                  "Sensitive data travels through the prompt"))
       + "        </div>\n",
  notes="Three gaps. Models are fluent but not exact. Tools, including MCP tools, are atomic calls with no "
        "declared dependencies and no provenance. And whenever tools must be combined, someone writes glue "
        "code, or the model improvises the pipeline in prose, step by step, with the data going through its prompt."))

# 3 ── definition ────────────────────────────────────────────────────────────────────
S.append(dict(file="slide-03-definition.html", h1="An agent-operated kernel",
  sub="A validated program whose intended operator is a language model with coding capabilities",
  foot="Validated, declared, composable, model-facing, traceable",
  body='        <div class="intro-card"><strong>KOAS</strong>, Kernel-Orchestrated Audit System: the name records the original intent, the audit of codebases of 10³–10⁵ classes. The concept is <strong>not limited to audit</strong>: the same contract now reads documents, reviews, translates, presents, scans for vulnerabilities and prepares responses.</div>\n'
       '        <div class="grid3">\n'
       + card("hgi-checkmark-badge-01", "Validated", desc("State-of-the-art method, defended by tests. Saqqara alone states <strong>154 propositions</strong>, each with a test."))
       + card("hgi-repeat", "Deterministic, or reproducible", desc("Compute kernels return the same output for the same input. Generative kernels pin model, decoding and prompt version."))
       + card("hgi-file-script", "Declared contract", desc("<code>name</code>, <code>version</code>, <code>stage</code>, <code>requires</code>, <code>provides</code>: what it needs and what it gives, machine-readable."))
       + card("hgi-ai-chat-02", "Model-facing", desc("Returns full data to a file and a <strong>compact summary</strong> for the model: the agent reads results, not the corpus."))
       + card("hgi-puzzle", "Composable without glue", desc("Kernels never call each other. The orchestrator chains them from their declared contracts."))
       + card("hgi-fingerprint-scan", "Traceable", desc("Every output carries kernel version, input hash, resolved dependencies and timing; runs leave an audit trail."))
       + "        </div>\n",
  notes="A word on the name first. KOAS stands for Kernel-Orchestrated Audit System, because code audit was the "
        "original intent. The concept outgrew it: the same contract now serves document reading, review, translation, "
        "presentation, security and response preparation. Then the definition, in six properties. Validated: a state-of-the-art method defended by tests. Deterministic "
        "where it computes, reproducible where it writes. A declared, machine-readable contract. Model-facing: a "
        "compact summary for the agent, the full data in a file. Composable with no glue code. And traceable, output by output."))

# 4 ── what it is not ────────────────────────────────────────────────────────────────
S.append(dict(file="slide-04-not.html", h1="What it is not",
  sub="The word kernel is taken several times; the neighbouring ideas each lack one property",
  foot="Not an OS kernel, not a tool call, not a workflow",
  body=table(["Neighbour", "What it is", "What an agent-operated kernel adds"], [
      ["OS / GPU / Jupyter kernel", "A scheduler, a compute routine, a language runtime", "Unrelated meaning: here a kernel is a validated analysis program"],
      ["Function or tool call", "One callable, one result", "Declared dependencies, a summary for the model, an output hash"],
      ["MCP tool", "A transport: how a model reaches a capability", "KOAS kernels are exposed through MCP, but are not defined by it"],
      ["Agent workflow / chain", "A pipeline written by a developer or improvised by the model", "The pipeline is computed by the orchestrator from contracts"],
      ["Scientific library", "Functions written for a human programmer", "An operator contract: stages, summaries, isolation levels, audit"],
  ]),
  notes="Where the concept sits. It is not an operating-system, GPU or Jupyter kernel. It is more than a tool "
        "call: it declares dependencies and returns a provenance. MCP is one of its doors, not its definition. Unlike "
        "an agent workflow, nobody writes the pipeline. And unlike a library, it is designed for a model operator."))

# 5 ── anatomy ───────────────────────────────────────────────────────────────────────
CODE = """<span class="k">class</span> Kernel(ABC):                 <span class="c"># ragix_kernels/base.py</span>
    name: str        = <span class="c">"base"</span>
    version: str     = <span class="c">"1.0.0"</span>
    category: str    = <span class="c">"base"</span>     <span class="c"># family</span>
    stage: int       = 0          <span class="c"># 1 collect · 2 analyse · 3 report</span>
    requires: List[str] = []      <span class="c"># kernels it depends on</span>
    provides: List[str] = []      <span class="c"># what it makes available</span>

    <span class="k">def</span> compute(self, input: KernelInput) -> Dict[str, Any]: ...
    <span class="k">def</span> summarize(self, data: Dict[str, Any]) -> str: ...

<span class="k">@dataclass</span>
<span class="k">class</span> KernelInput:
    workspace: Path
    config: Dict[str, Any]
    dependencies: Dict[str, Path]  <span class="c"># outputs of required kernels</span>"""
S.append(dict(file="slide-05-anatomy.html", h1="Anatomy: the Python contract",
  sub="Compact, expandable, general: two methods and six declared attributes",
  foot="Two methods, six attributes, one envelope",
  body='        <div class="row">\n'
       '          <div class="card-outer" style="flex:1.45">\n'
       '            <div class="card-header"><span class="icon-box"><i class="hgi-stroke hgi-source-code" aria-hidden="true"></i></span><span class="card-title">What a kernel author writes</span></div>\n'
       f'            <div class="card-inner"><pre class="code">{CODE}</pre></div>\n'
       '          </div>\n'
       + card("hgi-package", "What every run returns",
              desc("<code>KernelOutput</code>, the same envelope for all 101 kernels:")
              + pts("<code>success</code>, <code>data</code>, <code>summary</code>, <code>output_file</code>",
                    "<code>kernel_name</code>, <code>kernel_version</code>",
                    "<code>input_hash</code>, <code>execution_time_ms</code>",
                    "<code>dependencies_used</code>, <code>dependencies_resolved</code>",
                    "<code>warnings</code>, <code>errors</code>")
              + desc("A new kernel is a subclass plus its contract; <code>KernelRegistry.discover()</code> finds it."))
       + "        </div>\n",
  notes="The whole contract fits on one slide. A kernel declares six attributes and implements two methods: compute, "
        "which does the work, and summarize, which writes the compact text the model reads. Its input names a "
        "workspace, a configuration and the outputs of the kernels it requires. Every run returns the same envelope, "
        "with the kernel version, an input hash, timing and the dependencies actually used."))

# 6 ── two classes ───────────────────────────────────────────────────────────────────
S.append(dict(file="slide-06-classes.html", h1="Two classes of kernels",
  sub="Deterministic where they compute, reproducible where they write",
  foot="Exact numbers from compute kernels; pinned prose from generative ones",
  body='        <div class="row">\n'
       + card("hgi-calculator-01", "Compute kernels",
              desc("Audit, security, Saqqara, Tender, most of Docs and Summary.")
              + pts("No language model in the numeric path", "Same input, same output, byte for byte",
                    "Testable as ordinary software: propositions and fixtures",
                    "Example: <code>ast_scan</code>, <code>metrics</code>, <code>coupling</code>, <code>saqqara_index</code>"))
       + card("hgi-quill-write-01", "Generative kernels",
              desc("Reviewer, Translate, the authoring part of Summary.")
              + pts("The model produces content, so bit-exactness is not promised",
                    "Reproducibility instead: greedy decoding, pinned model digest, recorded prompt version",
                    "Calls cached and replayable", "Example: <code>translate_draft</code>, <code>translate_qa</code>"))
       + "        </div>\n",
  notes="Two classes. Compute kernels contain no model at all in their numeric path, so they are tested like ordinary "
        "software. Generative kernels use a local model to write, so they promise reproducibility instead of bit-exactness: "
        "greedy decoding, a pinned model digest and a recorded prompt version, with calls cached for replay."))

# 7 ── division of labour ────────────────────────────────────────────────────────────
S.append(dict(file="slide-07-roles.html", h1="Division of labour",
  sub="The agent decides the intent; the orchestrator builds the pipeline; kernels compute",
  foot="The model decides what; kernels compute how much; the record shows how",
  body=flow(("1 · Agent", "States the intent: which results, on which workspace. Names kernels or a stage."),
            ("2 · Orchestrator", "Resolves the dependency closure, orders it, distributes independent kernels to workers."),
            ("3 · Kernels", "Compute; write full data to files; return an envelope and a summary."),
            ("4 · Agent", "Reads summaries and metrics, interprets, writes, decides the next intent."))
       + '        <div class="row">\n'
       + card("hgi-ai-brain-01", "What the agent does", pts("Planning and intent", "Interpretation of summaries", "Prose around computed facts"))
       + card("hgi-flow-connection", "What the orchestrator does", pts("Manifest and registry", "Topological order, stages", "Worker pool, cache, audit trail"))
       + card("hgi-database", "What the agent never needs", pts("The raw corpus in its prompt", "Glue code between kernels", "To compute a number itself"))
       + "        </div>\n",
  notes="Roles are separated. The agent states an intent: which results it wants. The orchestrator turns that into a "
        "pipeline and distributes the work. Kernels compute and write their results to files. The agent then reads "
        "summaries and metrics, interprets them and decides the next step. The model never needs the corpus in its "
        "prompt, never writes glue code, and never computes a number itself."))

# 8 ── from intent to pipeline (example computed from the registry) ──────────────────
S.append(dict(file="slide-08-pipeline.html", h1="From intent to pipeline",
  sub="A real example, resolved by the RAGIX 0.75 registry",
  foot="Nobody wrote this pipeline: it follows from the contracts",
  body='        <div class="intro-card">The agent asks for three results: <code>hotspots</code>, <code>dead_code</code>, <code>coupling</code>. It names no dependency. <code>KernelRegistry.resolve_dependencies</code> returns <code>ast_scan → dependency → coupling → dead_code → metrics → hotspots</code>, and the orchestrator runs it in stages.</div>\n'
       '        <div class="card-outer"><div class="card-inner" style="gap:14px; justify-content:center">\n'
       '          <div class="lane"><span class="lbl">Asked</span><span class="chip asked">hotspots</span><span class="chip asked">dead_code</span><span class="chip asked">coupling</span></div>\n'
       '          <div class="lane"><span class="lbl">Stage 1</span><span class="chip">ast_scan</span><span class="arrow">&rarr;</span><span class="chip">dependency</span><span class="chip">metrics</span><span class="note">&nbsp;both need only ast_scan: run concurrently</span></div>\n'
       '          <div class="lane"><span class="lbl">Stage 2</span><span class="chip">coupling</span><span class="chip">dead_code</span><span class="chip">hotspots</span><span class="note">&nbsp;independent of each other: run concurrently</span></div>\n'
       '          <div class="lane"><span class="lbl">Declared</span><span class="note"><code>hotspots</code> requires <code>ast_scan, metrics, dependency</code> · <code>dead_code</code> requires <code>ast_scan, dependency</code> · <code>coupling</code> requires <code>dependency</code></span></div>\n'
       '        </div></div>\n',
  notes="A real example. The agent asks for hotspots, dead code and coupling, and names nothing else. The registry "
        "resolves the closure from the declared requirements: ast_scan first, then dependency and metrics, which run "
        "concurrently, then the three requested kernels, again concurrently. Nobody wrote this pipeline."))

# 9 ── cooperation without glue ──────────────────────────────────────────────────────
S.append(dict(file="slide-09-cooperation.html", h1="Cooperation without glue code",
  sub="Kernels cooperate through declared outputs in a shared workspace, never by calling each other",
  foot="Declare a contract, and the kernel joins every pipeline that needs it",
  body='        <div class="row">\n'
       + card("hgi-folder-01", "The workspace is the meeting point",
              pts("Each kernel writes its output file in the workspace",
                  "A dependent kernel receives those paths in <code>KernelInput.dependencies</code>",
                  "No shared mutable state between kernels"))
       + card("hgi-plug-socket", "Joining is declaring",
              pts("A new kernel declares <code>requires</code> and <code>provides</code>",
                  "The registry discovers it; the orchestrator chains it",
                  "No change to the kernels already there"))
       + card("hgi-refresh", "Reuse is free",
              pts("Outputs are cached; a stage can run with <code>use_cache</code>",
                  "An unchanged input is not recomputed",
                  "Summaries are collected for the model in one call"))
       + "        </div>\n",
  notes="Cooperation without glue code. Kernels never call each other: they meet in the workspace, through the files "
        "they declare. A new kernel joins by declaring what it requires and provides; the registry discovers it and the "
        "orchestrator chains it, with no change to existing kernels. Cached outputs make reuse free."))

# 10 ── cooperation, concurrently ────────────────────────────────────────────────────
S.append(dict(file="slide-10-concurrent.html", h1="Cooperation, concurrently",
  sub="Swarm-style execution: many kernels at once, each one isolated, all of them recorded",
  foot="Cooperation is the key; concurrency comes from the contracts",
  body='        <div class="row">\n'
       + card("hgi-flow-connection", "Concurrency from the graph",
              pts("Kernels whose requirements are met run together in a worker pool (<code>parallel</code>, <code>max_workers</code>)",
                  "The dependency graph, not a developer, decides what may run at once",
                  "Each kernel writes its own output: no locks inside kernels"))
       + card("hgi-share-08", "Sharing information",
              pts("Through declared outputs in the workspace, within a run",
                  "Through a memory store across runs and sessions: 18 <code>memory_*</code> MCP tools (write, search, recall, consolidate, link…)",
                  "Through summaries, the agent's view of every result"))
       + card("hgi-shield-key", "Still traceable",
              pts("Every concurrent output carries its own input hash and version",
                  "The audit trail records each kernel's run",
                  "Isolation levels apply to what each result exposes"))
       + "        </div>\n",
  notes="Cooperation is the key, and it runs concurrently. Whatever the dependency graph allows runs at the same time in "
        "a worker pool, and since each kernel only writes its own output, there are no locks to manage. Information is "
        "shared through declared outputs within a run, through a memory store across runs, and through summaries for the "
        "agent. Every concurrent result stays individually traceable."))

# 11 ── two doors ────────────────────────────────────────────────────────────────────
PYCODE = """<span class="k">from</span> ragix_kernels.registry <span class="k">import</span> KernelRegistry
KernelRegistry.discover()
for name <span class="k">in</span> KernelRegistry.resolve_dependencies([<span class="c">"hotspots"</span>]):
    out = KernelRegistry.get(name)().run(KernelInput(ws, cfg, deps))
    <span class="k">if</span> out.success: deps[name] = out.output_file
    <span class="k">else</span>: <span class="k">break</span>                  <span class="c"># inspect, branch, retry</span>"""
S.append(dict(file="slide-11-doors.html", h1="Two doors: Python and MCP",
  sub="One call per intent through MCP; full composition through Python",
  foot="MCP to reach the kernels; Python to compose them",
  body='        <div class="row">\n'
       '          <div class="card-outer" style="flex:1.3">\n'
       '            <div class="card-header"><span class="icon-box"><i class="hgi-stroke hgi-source-code" aria-hidden="true"></i></span><span class="card-title">Python: beyond one call</span></div>\n'
       f'            <div class="card-inner"><pre class="code">{PYCODE}</pre>'
       + desc("A coding agent writes this kind of loop: compose kernels, inspect envelopes, branch on results, retry, combine families. Illustrative; the orchestrator does the same with a manifest.")
       + '</div>\n          </div>\n'
       + card("hgi-plug-socket", "MCP: one call per intent",
              pts("30 <code>koas_*</code> tools: <code>koas_init</code>, <code>koas_run</code>, <code>koas_status</code>, <code>koas_summary</code>, plus family tools (audit, security, Saqqara, Tender, Translate)",
                  "18 <code>memory_*</code> tools for the shared memory",
                  "Any MCP client can drive the kernels; results come back as summaries and metrics"))
       + "        </div>\n",
  notes="Two doors. Through MCP, any client reaches the kernels with one call per intent: thirty koas tools and eighteen "
        "memory tools. Through Python, a coding agent goes further: it composes kernels, inspects each envelope, branches "
        "and retries, and combines families in ways no fixed tool list anticipates."))

# 12 ── sensitive data ───────────────────────────────────────────────────────────────
S.append(dict(file="slide-12-sensitive.html", h1="Sensitive data",
  sub="The agent can orchestrate without reading: what each result exposes is a declared level",
  foot="Orchestrate without reading",
  body=table(["Output level", "For", "Contains", "Excludes"], [
      ["INTERNAL", "The team running the analysis", "Full output with all metadata", "Nothing"],
      ["EXTERNAL", "Client delivery", "Redacted report, metrics, findings", "Paths, identifiers, internal metadata"],
      ["ORCHESTRATOR", "An external agent", "Metrics only", "All text content"],
      ["COMPLIANCE", "An auditor", "Full output and attestation", "Nothing; the trail is the point"],
  ])
       + '        <div class="intro-card"><code>ragix_kernels/output_sanitizer.py</code> applies the level to every result. For confidential corpora, <strong>RAGIX-Sealed</strong> (draft v0.1) goes further: originals sealed with AES-256-GCM, values replaced by role-aware placeholders and leak-scanned before any model sees them, with nine kernels of its own.</div>\n',
  notes="Sensitive data. Every result is emitted at a declared level: internal, external, orchestrator or compliance. At "
        "the orchestrator level an external agent gets metrics only, never text, so it can drive the analysis without "
        "reading the documents. For confidential corpora, RAGIX-Sealed, still a draft, seals the originals and only lets "
        "placeholdered, leak-scanned content reach any model."))

# 13 ── audit ────────────────────────────────────────────────────────────────────────
S.append(dict(file="slide-13-audit.html", h1="Everything can be audited",
  sub="Each output, each run and each model endpoint leaves a record",
  foot="Re-run it, or read the record: both give the same answer",
  body='        <div class="row">\n'
       + card("hgi-package", "Per output", pts("Kernel name and version", "Input hash", "Dependencies used and resolved", "Execution time, warnings, errors"))
       + card("hgi-file-validation", "Per run", pts("An audit trail, kernel by kernel (<code>ExecutionContext</code>)", "A Merkle root over the inputs, in canonical order (<code>merkle.py</code>)", "A centralised activity log"))
       + card("hgi-shield-01", "Per model call", pts("The endpoint is captured (sovereignty info)", "Calls cached and replayable", "Generative kernels record model digest and prompt version"))
       + "        </div>\n",
  notes="Everything can be audited. Each output carries its kernel version, input hash and dependencies. Each run leaves "
        "an audit trail and a Merkle root over its inputs, computed in canonical order. Each model call records its "
        "endpoint and is cached for replay. A result can be re-run or read back from the record."))

# 14 ── scale today ──────────────────────────────────────────────────────────────────
S.append(dict(file="slide-14-scale.html", h1="Scale today: RAGIX 0.75",
  sub="101 registered kernels in 9 families, counted from the registry: audit, the original intent, is now one of nine",
  foot="One contract across 101 kernels",
  body='        <div class="kpis"><div class="kpi"><div class="big">101</div><div class="lab">registered kernels</div></div><div class="kpi"><div class="big">9</div><div class="lab">families</div></div><div class="kpi"><div class="big">48</div><div class="lab">MCP tools (30 koas_*, 18 memory_*)</div></div><div class="kpi"><div class="big">+9</div><div class="lab">kernels in RAGIX-Sealed</div></div></div>\n'
       + table(["Family", "Kernels", "Verb"], [
           ["Audit", "27", "analyse code: AST, metrics, coupling, risk"],
           ["Docs", "17", "digest and summarise documents, with citations"],
           ["Reviewer", "13", "review and edit long documents, reversibly"],
           ["Summary", "12", "consolidate many sources, Graph-RAG"],
           ["Security", "11", "discover, assess, check compliance"],
           ["Presenter · Saqqara · Translate · Tender", "8 · 6 · 6 · 1", "decks · typed document trees · EN→FR · responses"],
       ]),
  notes="Where it stands. RAGIX 0.75 registers 101 kernels in nine families, with forty-eight MCP tools, and nine more "
        "kernels in RAGIX-Sealed. Every one of them follows the same contract."))

# 15 ── beyond RAGIX: generative simulation ─────────────────────────────────────────
S.append(dict(file="slide-15-gs.html", h1="Beyond RAGIX: Generative Simulation",
  sub="The same operating principle, applied to scientific simulation",
  foot="Say it. Simulate it with AI.",
  body='        <div class="intro-card"><strong>Generative Simulation</strong> (<code>github.com/ovitrac/generativeSimulation</code>): simulation engines are developed or wrapped so that generative AI can manipulate them directly, without intensive retraining or fine-tuning. The model does not hold the physics; it <strong>controls, audits and refines</strong> simulations through external kernels: the Language-First Computational Lab.</div>\n'
       '        <div class="row">\n'
       + card("hgi-atom-01", "Scientific kernels", pts("Validated methods: mass transfer, oxidation kinetics, molecular dynamics; Radigen and Pizza³ cite their peer-reviewed publications", "Structured physical quantities in; machine-readable and human-readable results out", "Compact Python interfaces and symbolic languages a model can write"))
       + card("hgi-ai-brain-01", "The model as operator", pts("Interprets the question (&ldquo;what if T rises by 10 °C?&rdquo;)", "Writes the Python or the domain script that drives the kernels", "Checks assumptions and flags ill-posed logic"))
       + card("hgi-link-02", "Shared with KOAS", pts("The model decides; the kernel computes", "Numbers come from validated code, not from the model", "Results are reproducible and can be checked"))
       + "        </div>\n",
  notes="Agent-operated kernels are not only a RAGIX idea. Generative Simulation applies the same principle to science: "
        "simulation engines built or wrapped so that a generative model can drive them directly, without retraining. "
        "The model does not hold the physics; it controls, audits and refines simulations through external kernels."))

# 16 ── scientific kernels ───────────────────────────────────────────────────────────
S.append(dict(file="slide-16-science.html", h1="Scientific kernels, operated the same way",
  sub="Open-source demonstrators of Generative Simulation, all on GitHub",
  foot="Different sciences, one way of operating them",
  body=table(["Kernel", "What it computes", "Repository"], [
      ["SFPPy", "Food-contact compliance and risk assessment by first-principles migration modelling (US FDA, EU, China GB lists), in a symbolic language AI models parse", "<code>github.com/ovitrac/SFPPy</code>"],
      ["Pizza³", "Multiscale modelling with LAMMPS in Python: objects compile to LAMMPS scripts through operators (+, |, *); built for LLM prototyping", "<code>github.com/ovitrac/Pizza3</code>"],
      ["Radigen", "Radical oxidation mechanisms in oils, biofuels and polymers, built from prompts; two peer-reviewed kinetic frameworks", "<code>github.com/ovitrac/radigen</code>"],
      ["sig2dna", "Analytical signals turned into a DNA-like symbolic code: alignment, classification, blind source separation", "<code>github.com/ovitrac/sig2dna</code>"],
      ["SFPPylite", "SFPPy in the browser, with JupyterLite: no installation, no server", "<code>github.com/ovitrac/SFPPylite</code>"],
      ["GS-Hub", "Agents for scientific and technological reasoning, shared between communities and LLMs", "<code>github.com/ovitrac/GSHub</code>"],
  ]),
  notes="The scientific kernels. SFPPy for food-contact compliance, Pizza cubed for multiscale molecular dynamics with "
        "LAMMPS, Radigen for oxidation kinetics, sig2dna for analytical signals, SFPPylite to run SFPPy in a browser, and "
        "GS-Hub for the agents. Each is a validated method behind an interface a model can drive, and each is public."))

# 17 ── next and closing ─────────────────────────────────────────────────────────────
S.append(dict(file="slide-17-next.html", h1="Next steps",
  sub="What the concept asks for next",
  foot="The model decides what; kernels compute how much; the record shows how",
  body='        <div class="row">\n'
       + card("hgi-user-group", "Several agents, one workspace", desc("Today one orchestrator distributes work to concurrent kernels. Next: several agents dispatching kernels on a shared workspace, with the same contracts and the same trail."))
       + card("hgi-route-01", "Intent closer to the goal", desc("Today the agent names kernels or a stage. Next: kernels that declare what they <em>provide</em> in a vocabulary the agent can query directly."))
       + card("hgi-checkmark-badge-01", "Validation everywhere", desc("Saqqara's proposition-and-test gate as the rule for every family; Sealed from draft to production (NER, persistence, signed grants)."))
       + "        </div>\n"
       + '        <div class="intro-card">Code and documentation: <strong>github.com/ovitrac/RAGIX</strong> — <code>ragix_kernels/base.py</code>, <code>registry.py</code>, <code>orchestrator.py</code>, <code>docs/KOAS.md</code>. Adservio Innovation Lab.</div>\n',
  notes="Next steps. Several agents dispatching kernels on one shared workspace. Intents expressed closer to the goal, "
        "by querying what kernels provide. And the validation gate used by Saqqara extended to every family. The code is "
        "open: base, registry and orchestrator are the three files to read first."))

# 18 ── links ────────────────────────────────────────────────────────────────────────
S.append(dict(file="slide-18-links.html", h1="Links",
  sub="Everything in this deck is open source and public on GitHub",
  foot="Read the code: every claim in this deck points to it",
  body=table(["Project", "Repository", "Start with"], [
      ["RAGIX · KOAS", "<code>github.com/ovitrac/RAGIX</code>", "<code>ragix_kernels/base.py</code>, <code>registry.py</code>, <code>orchestrator.py</code>, <code>docs/KOAS.md</code>"],
      ["Generative Simulation", "<code>github.com/ovitrac/generativeSimulation</code>", "the concept and the Language-First Computational Lab"],
      ["GS-Hub", "<code>github.com/ovitrac/GSHub</code>", "agents for scientific reasoning"],
      ["SFPPy · SFPPylite", "<code>github.com/ovitrac/SFPPy</code> · <code>github.com/ovitrac/SFPPylite</code>", "the compliance notebooks"],
      ["Pizza³", "<code>github.com/ovitrac/Pizza3</code>", "the online documentation"],
      ["Radigen · sig2dna", "<code>github.com/ovitrac/radigen</code> · <code>github.com/ovitrac/sig2dna</code>", "the core classes"],
  ])
       + '        <div class="intro-card">Olivier Vitrac, PhD, HDR · Adservio Innovation Lab</div>\n',
  notes="All the links. RAGIX and its KOAS kernels, Generative Simulation and GS-Hub, and the scientific kernels: SFPPy "
        "and SFPPylite, Pizza cubed, Radigen and sig2dna. Every one is public."))


def cover_html(total: int) -> str:
    return HEAD.format(title="Slide 01 — Agent-Operated Kernels") + f"""  <div class="container" style="padding:0; position:relative">
    <div class="cover">
      <h1>Agent-Operated Kernels</h1>
      <p class="sub">Validated programs operated by language models: declared contracts, pipelines computed by an orchestrator, concurrent cooperation, and an audit trail. The KOAS design in RAGIX.</p>
      <p class="koas"><strong>KOAS</strong> = <strong>K</strong>ernel-<strong>O</strong>rchestrated <strong>A</strong>udit <strong>S</strong>ystem. Audit was the original intent; the concept is not limited to it. Its scientific counterpart: Generative Simulation.</p>
      <div class="meta"><span>Technical briefing</span><span>RAGIX 0.75 · KOAS</span><span>November 2026</span><span>Open source · MIT</span></div>
      <div class="by">Olivier Vitrac, PhD, HDR · Adservio Innovation Lab</div>
    </div>
  </div>

  <script src="../library/slide-common.js"></script>
</body>
</html>
"""


def main() -> None:
    os.makedirs(SLIDES, exist_ok=True)
    total = len(S)
    for n, s in enumerate(S, 1):
        if s.get("cover"):
            html = cover_html(total)
        else:
            html = HEAD.format(title=f"Slide {n:02d} — {s['h1']}") + PAGE.format(
                h1=s["h1"], sub=s["sub"], body=s["body"].rstrip("\n"), foot=s["foot"], n=n, total=total)
        # Every repository named in a slide is a live link (it opens outside the deck).
        html = re.sub(r'<code>(github\.com/ovitrac/[A-Za-z0-9_.-]+)</code>',
                      r'<a href="https://\1" target="_blank" rel="noopener"><code>\1</code></a>', html)
        with open(os.path.join(SLIDES, s["file"]), "w", encoding="utf-8") as f:
            f.write(html)
    # The browsable viewer (kernels-deck/index.html) lists the same slides, in order.
    index = os.path.join(HERE, "kernels-deck", "index.html")
    viewer = open(index, encoding="utf-8").read()
    entries = ",\n".join(
        f"      {{ file: 'slides/{s['file']}', title: '{('Cover' if s.get('cover') else s['h1']).replace(chr(39), chr(8217))}' }}"
        for s in S)
    viewer, n = re.subn(r"(var VIEWER_SLIDES = \[\n)[\s\S]*?(\n\s*\];)", lambda m: m.group(1) + entries + m.group(2), viewer, count=1)
    assert n == 1, "VIEWER_SLIDES not found in kernels-deck/index.html"
    viewer = re.sub(r'<span id="fs-total">\d+</span>', f'<span id="fs-total">{total}</span>', viewer)
    viewer = re.sub(r'<span id="total-slides">\d+</span>', f'<span id="total-slides">{total}</span>', viewer)
    viewer = re.sub(r'src="slides/[^"]+" scrolling', f'src="slides/{S[0]["file"]}" scrolling', viewer)
    viewer = viewer.replace("RAGIX — Tech Sheet", "Agent-Operated Kernels")
    open(index, "w", encoding="utf-8").write(viewer)
    with open(os.path.join(HERE, "kernels-deck", "notes.json"), "w", encoding="utf-8") as f:
        json.dump([{"file": s["file"], "notes": s["notes"]} for s in S], f, indent=1, ensure_ascii=False)
    print(f"{total} slides -> {SLIDES}")


if __name__ == "__main__":
    main()
