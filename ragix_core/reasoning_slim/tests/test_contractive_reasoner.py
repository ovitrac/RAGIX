import json

import pytest

from ContractiveReasoner import ContractiveReasoner


@pytest.mark.asyncio
async def test_leaf_solving_without_decomposition(monkeypatch):
    engine = ContractiveReasoner(entropy_decompose_threshold=10.0, k_entropy_samples=1)

    async def fake_init(self):
        self._ctx_window = 2048

    async def fake_estimate(self, node):
        debug = {
            "all_samples": ["leaf"],
            "distinct": 1,
            "freq": {"leaf": 1},
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        return 0.05, 0.05, 0.0, "leaf answer", debug

    monkeypatch.setattr(engine, "_init_model_info", fake_init.__get__(engine, ContractiveReasoner))
    monkeypatch.setattr(
        engine,
        "_estimate_entropy_and_answer",
        fake_estimate.__get__(engine, ContractiveReasoner),
    )

    res = await engine.solve("simple question", max_depth=2, max_loops=2)
    assert res.final_answer == "leaf answer"
    summary = res.summarize()
    assert summary["total_nodes"] == 1
    assert summary["states"].get("solved") == 1


class FakeReasoner(ContractiveReasoner):
    """
    Deterministic stub reasoner to exercise decomposition/collapse without network calls.
    """

    def __init__(self):
        super().__init__(entropy_decompose_threshold=0.2, k_entropy_samples=1)
        self._ctx_window = 1024

    async def _init_model_info(self) -> None:
        self._ctx_window = 1024

    async def _estimate_entropy_and_answer(self, node):
        depth = self._node_depth(node.node_id)
        if depth == 0:
            entropy = 1.2  # force decomposition
            answer = "root candidate"
        else:
            entropy = 0.05  # force solve
            answer = f"child-{depth}"
        debug = {
            "all_samples": [answer],
            "distinct": 1,
            "freq": {answer: 1},
            "token_usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return entropy, 0.05, 0.0, answer, debug

    async def _chat(self, messages, temperature: float = 0.3):
        user_content = messages[-1]["content"]
        if "DECOMPOSE" in user_content:
            resp = {
                "content": json.dumps(
                    {
                        "subquestions": [
                            {"type": "AND", "question": "child q1", "role": "analysis"},
                            {"type": "AND", "question": "child q2", "role": "analysis"},
                        ]
                    }
                ),
                "prompt_eval_count": 2,
                "eval_count": 1,
            }
        elif "COLLAPSE" in user_content:
            resp = {"content": "collapsed final", "prompt_eval_count": 1, "eval_count": 1}
        elif "FINAL_SUMMARY" in user_content:
            resp = {"content": "fallback", "prompt_eval_count": 1, "eval_count": 1}
        else:
            resp = {"content": "leaf solved", "prompt_eval_count": 1, "eval_count": 1}
        self._record_token_usage(resp)
        return resp


@pytest.mark.asyncio
async def test_decompose_and_collapse_flow():
    engine = FakeReasoner()
    res = await engine.solve("root question", max_depth=3, max_loops=4)
    summary = res.summarize()

    assert res.final_answer in ("collapsed final", "fallback")
    assert summary["total_nodes"] == 3
    assert summary["states"].get("solved") == 3
    assert summary["depth"]["max"] == 1

    trace = res.export_trace()
    assert len(trace["nodes"]) == 3
    assert "analysis" in res.to_mermaid().lower()


@pytest.mark.asyncio
async def test_trace_and_summary_tokens_accumulate():
    engine = FakeReasoner()
    res = await engine.solve("another question", max_depth=2, max_loops=3)
    summary = res.summarize()
    trace = res.export_trace()

    assert summary["tokens"]["prompt"] > 0
    assert summary["tokens"]["completion"] > 0
    # ensure metrics were serialized
    assert all("metrics" in node for node in trace["nodes"].values())
