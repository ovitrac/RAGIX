import json
from pathlib import Path

from benchmarks.summarize_benchmarks import summarize, _flatten_summary


def test_flatten_summary_extracts_metrics(tmp_path: Path):
    sample = {
        "question_id": "q1",
        "scenario": "s1",
        "duration_sec": 1.2,
        "final_answer_chars": 42,
        "summary": {
            "total_nodes": 3,
            "steps": 4,
            "depth": {"max": 2},
            "tokens": {"prompt": 10, "completion": 5},
            "entropies": {
                "model": {"mean": 0.5},
                "struct": {"mean": 0.2},
                "consistency": {"mean": 0.1},
            },
            "relevance_root": {"mean": 0.8},
        },
    }
    flat = _flatten_summary(sample)
    assert flat["question_id"] == "q1"
    assert flat["scenario"] == "s1"
    assert flat["nodes"] == 3
    assert flat["depth_max"] == 2
    assert flat["entropy_model_mean"] == 0.5
    assert flat["relevance_root_mean"] == 0.8


def test_summarize_reads_directory(tmp_path: Path):
    data = {
        "question_id": "q1",
        "scenario": "s1",
        "duration_sec": 1.0,
        "final_answer_chars": 10,
        "summary": {
            "total_nodes": 2,
            "steps": 2,
            "depth": {"max": 1},
            "tokens": {"prompt": 3, "completion": 2},
            "entropies": {
                "model": {"mean": 0.3},
                "struct": {"mean": 0.1},
                "consistency": {"mean": 0.0},
            },
            "relevance_root": {"mean": 0.5},
        },
    }
    path = tmp_path / "q1_s1_summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    rows = summarize(tmp_path)
    assert len(rows) == 1
    assert rows[0]["nodes"] == 2
