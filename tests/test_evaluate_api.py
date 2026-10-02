"""Phase 9 — POST /evaluate (memory vs no-memory comparison)."""

from fastapi.testclient import TestClient

from tests.conftest import hash_embed


def _client(api_app):
    return TestClient(api_app.app)


def _seed(real_memory, content, agent="alice"):
    real_memory.store.store(
        agent_id=agent, content=content, embedding=hash_embed(content),
        context={"role": "user"},
    )


class TestEvaluateEndpoint:

    def test_reports_memory_use_and_relevance(
        self, api_app, real_memory, auth_headers
    ):
        _seed(real_memory, "alice grows tomatoes in her garden")
        r = _client(api_app).post("/evaluate", headers=auth_headers, json={
            "agent_id": "alice",
            "test_questions": ["what do I grow in my garden?"],
        })
        assert r.status_code == 200
        body = r.json()
        res = body["results"][0]
        assert res["memories_used"] >= 1
        assert res["maem_used_memory"] is True
        assert res["memory_relevance"] >= 0.20
        assert body["mean_relevance"] >= 0.20
        assert body["maem_accuracy"] is None  # no expected answers given

    def test_unrelated_question_uses_no_memory(
        self, api_app, real_memory, auth_headers
    ):
        _seed(real_memory, "alice grows tomatoes in her garden")
        r = _client(api_app).post("/evaluate", headers=auth_headers, json={
            "agent_id": "alice",
            "test_questions": ["capital of france?"],
        })
        res = r.json()["results"][0]
        assert res["memories_used"] == 0
        assert res["memory_relevance"] is None

    def test_answer_accuracy_scored_against_expected(
        self, api_app, auth_headers
    ):
        # the shell orchestrator's LLM always answers "Okay, noted."
        r = _client(api_app).post("/evaluate", headers=auth_headers, json={
            "agent_id": "alice",
            "test_questions": ["q one", "q two"],
            "expected_answers": ["noted", "Ravi|Ravi Kumar"],
        })
        body = r.json()
        assert [x["maem_correct"] for x in body["results"]] == [True, False]
        assert body["maem_accuracy"] == 0.5
        assert body["baseline_accuracy"] == 0.5

    def test_expected_answers_length_mismatch_is_400(
        self, api_app, auth_headers
    ):
        r = _client(api_app).post("/evaluate", headers=auth_headers, json={
            "agent_id": "alice",
            "test_questions": ["a", "b"],
            "expected_answers": ["only one"],
        })
        assert r.status_code == 400

    def test_baseline_rate_is_reported_not_hardcoded(
        self, api_app, auth_headers
    ):
        r = _client(api_app).post("/evaluate", headers=auth_headers, json={
            "agent_id": "alice", "test_questions": ["anything"],
        })
        # "Okay, noted." contains no "don't know"-style phrase
        assert r.json()["baseline_recall_rate"] == 1.0
