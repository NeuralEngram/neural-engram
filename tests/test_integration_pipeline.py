# tests/test_integration_pipeline.py
"""
End-to-end: Orchestrator.route() → PlannerAgent.plan() →
ExecutorAgent.execute() → memory stored in all three tiers → the next
route() for the same agent_id gets that memory back.

Everything is the real object graph built by Orchestrator.__init__
(see conftest.real_orchestrator). Assertions are made on what the fake
Anthropic client *received* — i.e. what would really have been sent to
the API — and on what the real memory tiers hold afterwards.

Guards all three original scoping bugs:
  #1 working memory not scoped per agent   → test_second_turn_*_conversation
  #2 executor read its own WM bucket       → test_second_turn_*_conversation
  #3 planner skipped semantic-sourced ids  → test_prior_turn_memories_reinforced
"""

import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from memory.semantic_memory import SemanticMemory
from tests.conftest import TEST_API_KEY


TURN_1 = "my son Arjun visits on sundays"
TURN_2 = "which day does Arjun visit?"
REPLY = "Okay, noted."


def contents(messages):
    return [m["content"] for m in messages]


# ─── Single turn ───────────────────────────────────────────

class TestFirstTurn:

    def test_request_sent_to_anthropic(self, real_orchestrator):
        orch = real_orchestrator()

        result = orch.route(TURN_1, agent_id="rajan")

        assert result["response"] == REPLY
        assert len(orch.client.requests) == 1
        req = orch.client.requests[0]
        assert req["messages"] == [{"role": "user", "content": TURN_1}]
        assert req["model"]
        assert "Relevant memory context" not in req["system"]

    def test_turn_stored_in_every_tier_for_that_agent(self, real_orchestrator):
        orch = real_orchestrator()
        mo = orch.memory_store.orchestrator

        orch.route(TURN_1, agent_id="rajan")

        assert contents(orch.memory_store.get_context("rajan")) == [
            TURN_1, REPLY
        ]
        episodes = mo.episodic_memory.get_all()
        assert sorted(ep.content for ep in episodes) == sorted([TURN_1, REPLY])
        for ep in episodes:
            assert ep.agent_id == "rajan"
            sem = mo.semantic_memory.get(ep.episode_id)
            assert sem is not None
            assert sem.metadata["agent_id"] == "rajan"

    def test_nothing_lands_in_internal_agent_buckets(self, real_orchestrator):
        orch = real_orchestrator()

        orch.route(TURN_1, agent_id="rajan")

        wm = orch.memory_store.orchestrator.working_memory
        for bucket in ("default", orch.planner.agent_id,
                       orch.executor.agent_id):
            assert wm.is_empty(bucket), bucket

    def test_result_shape(self, real_orchestrator):
        orch = real_orchestrator()

        result = orch.route("remember my son Arjun", agent_id="rajan")

        assert result["intent"] == "memory_retrieval"
        assert result["needs_memory"] is True
        for key in ("planner_memories", "retrieved_memories",
                    "augmented_context", "task"):
            assert key in result

    def test_question_turn_not_stored_as_memory(self, real_orchestrator):
        orch = real_orchestrator()

        orch.route(TURN_2, agent_id="rajan")

        stored = [ep.content for ep in
                  orch.memory_store.orchestrator.episodic_memory.get_all()]
        assert TURN_2 not in stored
        assert REPLY in stored


# ─── Two turns: prior context comes back ───────────────────

class TestSecondTurn:

    def test_second_turn_gets_first_turn_as_conversation(
        self, real_orchestrator
    ):
        orch = real_orchestrator()

        orch.route(TURN_1, agent_id="rajan")
        orch.route(TURN_2, agent_id="rajan")

        assert orch.client.last_messages == [
            {"role": "user", "content": TURN_1},
            {"role": "assistant", "content": REPLY},
            {"role": "user", "content": TURN_2},
        ]

    def test_second_turn_gets_first_turn_as_retrieved_memory(
        self, real_orchestrator
    ):
        orch = real_orchestrator()

        orch.route(TURN_1, agent_id="rajan")
        result = orch.route(TURN_2, agent_id="rajan")

        assert f"Memory: {TURN_1}" in orch.client.last_system
        assert TURN_1 in [
            m.content for m in result["retrieved_memories"]["fused_memory"]
        ]

    def test_each_memory_appears_once_in_prompt(self, real_orchestrator):
        orch = real_orchestrator()

        orch.route(TURN_1, agent_id="rajan")
        orch.route(TURN_2, agent_id="rajan")

        assert orch.client.last_system.count(f"Memory: {TURN_1}") == 1

    def test_prior_turn_memories_reinforced(self, real_orchestrator):
        """
        The relevant turn-1 memory (the user's own words) is retrieved
        and reinforced once. The turn-1 assistant reply is NOT
        retrieved (assistant replies are excluded from long-term
        retrieval by default, Phase 9) and so is not reinforced.
        """
        orch = real_orchestrator()
        episodic = orch.memory_store.orchestrator.episodic_memory
        semantic = orch.memory_store.orchestrator.semantic_memory

        orch.route(TURN_1, agent_id="rajan")
        by_content = {ep.content: ep.episode_id for ep in episodic.get_all()}
        assert set(by_content) == {TURN_1, REPLY}
        result = orch.route(TURN_2, agent_id="rajan")

        fused = result["planner_memories"]["fused_memory"]
        assert [m.content for m in fused] == [TURN_1]

        user_id = by_content[TURN_1]
        ep = episodic.get_by_id(user_id)
        assert ep.review_count == 1
        assert ep.agent_feedback == pytest.approx(0.65)
        assert semantic.get(user_id).reinforcement_count == 1

        reply = episodic.get_by_id(by_content[REPLY])
        assert reply.review_count == 0
        assert semantic.get(by_content[REPLY]).reinforcement_count == 0

    def test_assistant_replies_retrievable_when_enabled(
        self, real_orchestrator, monkeypatch
    ):
        monkeypatch.setenv("RETRIEVE_ASSISTANT_REPLIES", "true")
        orch = real_orchestrator()
        orch.route("rajan likes noted things", agent_id="rajan")
        result = orch.route("anything noted?", agent_id="rajan")
        assert REPLY in [
            m.content for m in result["planner_memories"]["fused_memory"]
        ]

    def test_long_conversation_keeps_order_and_window(self, real_orchestrator):
        orch = real_orchestrator()
        turns = [f"fact number {n} is recorded" for n in range(7)]

        for t in turns:
            orch.route(t, agent_id="rajan")

        sent = orch.client.last_messages
        # get_context() default window is 10 messages = previous 5 turns
        expected_history = []
        for t in turns[1:6]:
            expected_history += [
                {"role": "user", "content": t},
                {"role": "assistant", "content": REPLY},
            ]
        assert sent == expected_history + [
            {"role": "user", "content": turns[6]}
        ]


# ─── Through the HTTP API with a real Orchestrator ─────────

@pytest.fixture
def api_client(real_orchestrator, monkeypatch):
    import dotenv
    import orchestrator.orchestrator as orch_module

    orch = real_orchestrator()
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("MAEM_API_KEY", TEST_API_KEY)
    monkeypatch.setattr(orch_module, "Orchestrator", lambda: orch)
    sys.modules.pop("api.main", None)
    module = importlib.import_module("api.main")
    module.limiter.reset()

    yield TestClient(module.app), orch

    sys.modules.pop("api.main", None)


class TestThroughApi:

    H = {"X-API-Key": TEST_API_KEY}

    def test_chat_user_id_becomes_agent_id(self, api_client):
        client, orch = api_client

        client.post("/chat", json={"message": TURN_1, "user_id": "rajan"},
                    headers=self.H)
        resp = client.post("/chat", json={"message": TURN_2,
                                          "user_id": "rajan"}, headers=self.H)

        assert resp.status_code == 200
        assert resp.json()["memories_retrieved"] >= 1
        assert {"role": "user", "content": TURN_1} in orch.client.last_messages

    def test_context_endpoint_reflects_real_working_memory(self, api_client):
        client, _ = api_client

        client.post("/chat", json={"message": TURN_1, "user_id": "rajan"},
                    headers=self.H)

        ctx = client.get("/context/rajan", headers=self.H).json()
        assert contents(ctx) == [TURN_1, REPLY]
        assert client.get("/context/alice", headers=self.H).json() == []


# ─── Forgetting is real ────────────────────────────────────

class TestForgottenMeansForgotten:
    """
    Regression: apply_decay() removed forgotten episodes from RAM/SQLite
    but never from Qdrant, and the following cleanup() then had nothing
    left to prune — so on the scheduler path a "forgotten" memory kept
    coming back via semantic search into the very next LLM prompt.
    """

    def _age_everything(self, orch, days=90):
        mo = orch.memory_store.orchestrator
        with mo.episodic_lock:
            for ep in mo.episodic_memory.get_all():
                ep.last_reviewed_at -= __import__("datetime").timedelta(
                    days=days)
                mo.episodic_memory._save_episode(ep)

    @pytest.mark.parametrize("how", ["scheduler", "decay", "decay+cleanup"])
    def test_forgotten_memory_not_in_next_prompt(self, real_orchestrator, how):
        orch = real_orchestrator()
        orch.route("my daughter Priya brings my medicine", agent_id="rajan")
        self._age_everything(orch)

        if how == "scheduler":
            orch.scheduler.run_once()
        else:
            orch.memory_store.apply_decay()
            if how == "decay+cleanup":
                orch.memory_store.cleanup()

        mo = orch.memory_store.orchestrator
        assert mo.episodic_memory.get_all() == []
        assert mo.semantic_memory.count() == 0

        # a fresh conversation turn (not about Priya) must not see it
        orch.route("tell me something new", agent_id="rajan")
        assert "Priya" not in orch.client.last_system


class TestShutdown:

    def test_stop_closes_vector_store_and_is_idempotent(
        self, real_orchestrator
    ):
        orch = real_orchestrator()
        orch.route(TURN_1, agent_id="rajan")

        orch.stop()
        orch.stop()  # second call (e.g. fixture teardown) must not raise

        assert orch.memory_store.orchestrator.semantic_memory.client is None
        assert not orch.scheduler.is_running()

    def test_on_disk_store_reopens_after_close(self, tmp_path):
        from memory.semantic_memory import SemanticMemoryStore
        from tests.conftest import hash_embed

        path = str(tmp_path / "qdrant")
        first = SemanticMemoryStore(storage_path=path)
        first.add("11111111-1111-1111-1111-111111111111", "kept",
                  hash_embed("kept"))
        first.close()

        second = SemanticMemoryStore(storage_path=path)
        assert second.get("11111111-1111-1111-1111-111111111111").content \
            == "kept"
        second.close()
