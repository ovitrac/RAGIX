# Mapping: VirtualHybridLab → Code Audit System

```text

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  VIRTUAL HYBRID LAB                    →    CODE AUDIT LAB                  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  Scientific Kernels                    →    Audit Kernels                   │
  │  (FEniCS, LAMMPS, SFPPy, Pizza³)            (ragix-ast, partitioner, etc.)  │
  │                                                                             │
  │  Scientific Agent                      →    Audit Agent                     │
  │  (narrow, operates one kernel)              (operates one audit kernel)     │
  │                                                                             │
  │  Orchestrator Agent                    →    Audit Orchestrator              │
  │  (coordinates, schedules, diagnoses)        (stages, dependencies, flow)    │
  │                                                                             │
  │  Discovery Agent                       →    Audit Discovery                 │
  │  ("Interpret these experiments")            ("What are the risks here?")    │
  │                                                                             │
  │  Knowledge Base (RAG)                  →    Audit KB (RAG)                  │
  │  (regulatory data, domain knowledge)        (patterns, standards, history)  │
  └─────────────────────────────────────────────────────────────────────────────┘

  Architecture: Code Audit Lab (following GS pattern)

                      ┌──────────────────────────────────────┐
                      │      🔍 DISCOVERY AGENT              │
                      │  "What technical debt should we      │
                      │   prioritize? What are the risks?"   │
                      │                                      │
                      │  LLM: Claude/GPT-4 (large context)   │
                      └──────────────────┬───────────────────┘
                                         │
                      ┌──────────────────▼───────────────────┐
                      │      🧠 ORCHESTRATOR AGENT           │
                      │  Coordinates audit workflow          │
                      │  Handles dependencies & scheduling   │
                      │                                      │
                      │  LLM: Mistral (coordination)         │
                      │  Kernel: workflow_engine             │
                      └──────────────────┬───────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
          ▼                              ▼                              ▼
  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
  │  📊 METRICS AGENT │    │  🧩 STRUCTURE     │    │  📈 QUALITY AGENT │
  │                   │    │     AGENT         │    │                   │
  │  Kernel:          │    │  Kernel:          │    │  Kernel:          │
  │  • ragix-ast      │    │  • partitioner    │    │  • hotspot_detect │
  │  • metrics_compute│    │  • dep_graph      │    │  • dead_code      │
  │  • tech_debt      │    │  • service_detect │    │  • coupling       │
  │                   │    │                   │    │                   │
  │  LLM: Mistral     │    │  LLM: Mistral     │    │  LLM: Mistral     │
  │  (Python-savvy)   │    │  (Python-savvy)   │    │  (Python-savvy)   │
  └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
            │                        │                        │
            ▼                        ▼                        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        AUDIT KERNELS                                │
  │  (Pure computation - no LLM needed inside)                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │  ast_scan.py      → Wraps: ragix-ast scan                           │
  │  metrics.py       → Wraps: ragix-ast metrics                        │
  │  dependency.py    → Wraps: ragix-ast graph + build_dependency_graph │
  │  partitioner.py   → Wraps: ragix_audit/partitioner.py               │
  │  service_detect.py→ Wraps: ragix_audit/service_detector.py          │
  │  hotspots.py      → Wraps: ragix-ast hotspots                       │
  │  dead_code.py     → Wraps: partitioner dead_code detection          │
  │  coupling.py      → Computes Martin metrics (Ca, Ce, I, A, D)       │
  │  stats.py         → Aggregates all metrics into summary             │
  └─────────────────────────────────────────────────────────────────────┘
            │                        │                        │
            ▼                        ▼                        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    AUDIT KNOWLEDGE BASE                             │
  │  /path/to/audit/{PROJECT}/                                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │  stage1/          → Raw kernel outputs (JSON)                       │
  │  stage2/          → Analysis summaries (JSON + summary.txt)         │
  │  stage3/          → Report sections (Markdown)                      │
  │  kb/              → Indexed findings for RAG retrieval              │
  │  logs/            → Full audit trail (reproducibility)              │
  └─────────────────────────────────────────────────────────────────────┘

```

## Key Insight from VirtualHybridLab

  "Each interaction costs a few seconds; most time is spent inside the scientific kernels, not the LLMs."

  Applied to code audit:
  - Kernels do the heavy computation (AST parsing, metrics, graph analysis)
  - LLM orchestrates and interprets (what to run next, what does this mean)
  - RAG provides context (patterns, standards, previous audits)

  Proposed Kernel Interface (GS-style)

```text
# kernels/base.py - Following GS pattern
  from abc import ABC, abstractmethod
  from dataclasses import dataclass
  from pathlib import Path
  from typing import Dict, Any, Optional

  @dataclass
  class KernelOutput:
      """Standard output from any audit kernel."""
      success: bool
      data: Dict[str, Any]           # Full structured data
      summary: str                    # < 500 chars for LLM (decoder output)
      output_file: Path              # Persisted JSON
      dependencies_used: list[str]   # What stage1/stage2 files were read

  class AuditKernel(ABC):
      """
      Base class for all audit kernels.
      Following GS VirtualHybridLab pattern:
      - Kernel does computation (no LLM inside)
      - Produces structured output + summary for LLM consumption
      - Fully traceable and reproducible
      """

      name: str                      # Kernel identifier
      stage: int                     # 1, 2, or 3
      dependencies: list[str]        # Required kernel outputs
    
      @abstractmethod
      def run(self, workspace: Path, config: Dict) -> KernelOutput:
          """Execute kernel computation."""
          pass
    
      def encode_for_llm(self, output: KernelOutput) -> str:
          """
          Upstream RAG: Encode output for LLM consumption.
          Returns contextualized summary (< 500 chars).
          """
          return output.summary
    
      def decode_from_llm(self, llm_response: str) -> Dict:
          """
          Downstream RAG: Decode LLM interpretation back to structured form.
          Used when LLM provides guidance for next steps.
          """
          # Parse LLM response into actionable structure
          pass

  Example: Wrapping RAGIX Tools as Kernels

  # kernels/stage1/ast_scan.py
  """
  Kernel: AST Scan
  Wraps: ragix-ast scan
  """
  import subprocess
  import json
  from pathlib import Path
  from ..base import AuditKernel, KernelOutput

  class ASTScanKernel(AuditKernel):
      name = "ast_scan"
      stage = 1
      dependencies = []  # No dependencies - first kernel

      def run(self, workspace: Path, config: Dict) -> KernelOutput:
          project_path = config["project"]["path"]
          language = config["project"]["language"]
    
          # Call existing RAGIX tool (reuse, don't reimplement!)
          result = subprocess.run(
              ["ragix-ast", "scan", project_path, "--lang", language, "--json"],
              capture_output=True, text=True
          )
    
          data = json.loads(result.stdout)
    
          # Generate summary for LLM (encoder output)
          summary = (
              f"Scanned {data['total_files']} {language} files. "
              f"Found {data['total_classes']} classes, {data['total_methods']} methods, "
              f"{data['total_loc']:,} LOC. "
              f"Parse errors: {data.get('errors', 0)}."
          )
    
          # Persist output
          output_file = workspace / "stage1" / "ast_scan.json"
          output_file.parent.mkdir(parents=True, exist_ok=True)
          output_file.write_text(json.dumps(data, indent=2))
    
          return KernelOutput(
              success=result.returncode == 0,
              data=data,
              summary=summary,
              output_file=output_file,
              dependencies_used=[]
          )

```