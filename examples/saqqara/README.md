# saqqara — a demo you can run on a clean clone

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Four scripts, about a minute, no network and no model download.

```bash
pip install -e ".[saqqara,dev]"
bash examples/saqqara/run_demo.sh            # into a temporary workspace
bash examples/saqqara/run_demo.sh ./my-work  # or a directory you choose
```

**The documents are built by the scripts**, using the same generators the gates
use. No document is committed to this repository — not one — so everything the
demo prints came from code you can read, and the guard refuses a binary appearing
under `examples/saqqara/`.

## What each step shows

| script | what it demonstrates |
|---|---|
| `01_read_tree.py` | five formats into typed trees; one node with its full citation chain; the two roots and how they differ |
| `02_index_store.py` | chunking along the tree into one SQLite file; what was refused and why, per document |
| `03_search_with_trace.py` | both lanes, each rank kept separately, every hit walked back to the nodes it came from |
| `04_lexical_only.py` | two refusals worth seeing: no zero vectors without an embedder, and a configuration typo refused with its path |

## The dense lane is optional

`saqqara.yaml` sets `embedder.provider: none`, so the demo runs lexical-only and
says so. `03_search_with_trace.py` turns the dense lane on if
`sentence-transformers` is importable and **skips it with a printed reason** if it
is not — a demo that cannot run without a model download is a demo nobody runs.

To see both lanes:

```bash
pip install -e ".[saqqara,retrieval]"
```

## What to look at

- **Every hit names its nodes**, and every node names its source, format and
  position. A hit that could not do that would be text with a score.
- **The refusal counts are the point**, not noise. Every node is chunked,
  absorbed by a node that was, or refused with a reason — nothing is dropped
  quietly.
- **`dense: disabled (no embedder)`** is stated rather than left to be inferred.
  Nothing writes a zero vector to make the column look populated.
