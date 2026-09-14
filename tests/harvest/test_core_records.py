"""What pass 1 actually writes to disk, driven end to end with no model (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Two defects of the same family are what this test exists for, both found by reading a record rather
than the code that makes it: a sheet's `caveat` was assigned to the returned dict **after** `call` had
already serialised it, so the JSONL never carried what the docstring promised; and a parent's assembled
source was recorded nowhere, so a document's card could be audited against nothing. Both are now
written with the record, and both are asserted **from the file**, never from the in-memory object —
an assertion on the dict would have passed happily while the run was losing the field.

The driver is exercised whole against the immutable store, with `get` and `post` replaced by canned
answers: no Ollama, no GPU, no network. The piece is one small workbook with three sheet roll-ups, two
of which hold no window — so one sheet is summarised from its children's abstracts (the assembled
path) and the others from the store text (the store path), and the document above them assembles from
the sheets.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PASS1 = HERE.parent / "demoE2E/verify/pass1"
#: the reference store is local and never committed; its path comes from the environment, and
#: every test that needs it is skipped where it is absent
STORE = Path(os.environ.get("HARVEST_REFERENCE_STORE", "/nonexistent/saqqara.db"))
PIECE = ("CCTP REF-1 - Annexe 4 - Fichiers de quantification adhérent/Catalogue des besoins/"
         "Catalogue des besoins (12) Equipements de manutention non medicaux et engins (v2).xlsx")
ANSWER = ("Le lot couvre les engins de manutention non médicaux. "
          "La maintenance préventive est annuelle.")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    if not STORE.is_file():
        pytest.skip("the store is not on this host")
    from ragix_kernels.harvest import core

    out = tmp_path_factory.mktemp("pass1")
    paths = out / "paths.txt"
    paths.write_text(PIECE + "\n", encoding="utf-8")

    def fake_get(host, path, timeout=20.0):
        if path == "/api/tags":
            return {"models": [{"name": core.MODEL, "digest": core.DIGEST + "0" * 8}]}
        if path == "/api/ps":
            return {"models": []}
        return {}

    def fake_post(host, path, body, timeout):
        return {"response": ANSWER, "done_reason": "stop", "prompt_eval_count": 120, "eval_count": 40}

    core.get, core.post = fake_get, fake_post
    code = core.main(["--store", str(STORE), "--lab", str(HERE.parent), "--out", str(out),
                      "--select", f"paths:{paths}", "--sheets-as-nodes", "--expect", "5,1"])
    records = [json.loads(line) for line
               in (out / "core_abstracts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return code, records


def test_the_run_completes(run):
    code, records = run
    assert code == 0
    assert records, "no record was written"


def test_every_sheet_record_carries_its_caveat_on_disk(run):
    """The defect: the key existed on the returned dict and never in the file.

    The expected text is written out here as a literal rather than imported from `core`, so that the
    assertion is about what reached the file. Imported, the test failed on a missing attribute when
    the fix was absent — which would also have passed the day someone declared the constant and still
    wrote the record without it.
    """
    sheets = [r for r in run[1] if r.get("level") == "sheet"]
    assert sheets, "the fixture produced no sheet record"
    missing = [r["node_id"][:16] for r in sheets
               if "overlap by 200 characters at each seam" not in (r.get("caveat") or "")]
    assert not missing, f"sheet records written without the caveat: {missing}"


def test_an_assembled_node_records_what_it_was_written_from(run):
    """A parent must carry its recipe, the children it used, and their abstract hashes."""
    assembled = [r for r in run[1] if r.get("assembly")]
    assert assembled, "no node recorded an assembly"
    for record in assembled:
        used, hashes = record.get("children_used"), record.get("children_sha256")
        assert used and hashes and len(used) == len(hashes), record["node_id"]
        assert set(used) <= set(record.get("children") or []), "a child was used that was never offered"


def test_the_audit_can_rebuild_every_parent(run):
    """The point of the provenance: `rebuild_parent` reproduces the source from the run alone."""
    from ragix_kernels.harvest import pass1 as P

    by_id = {r["node_id"]: r for r in run[1]}
    rebuilt = [r for r in run[1] if r.get("assembly") and P.rebuild_parent(r, by_id)]
    assert rebuilt, "no parent could be rebuilt from its children"


def test_a_changed_child_abstract_refuses(run):
    """And the direction that matters: the verifier must not accept a parent whose child moved."""
    from ragix_kernels.harvest import pass1 as P

    parent = next(r for r in run[1] if r.get("assembly"))
    by_id = {r["node_id"]: r for r in run[1]}
    child = by_id[parent["children_used"][0]]
    by_id[child["node_id"]] = {**child, "abstract": (child.get("abstract") or "") + " Ajout."}
    with pytest.raises(P.ParentDrift):
        P.rebuild_parent(parent, by_id)
