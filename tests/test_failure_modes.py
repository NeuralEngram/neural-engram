# tests/test_failure_modes.py
"""
Forced failures in the live pipeline (real Orchestrator, fake Anthropic).

The contract in ExecutorAgent.execute():
  - LLM failure            → propagates; the request fails; nothing of
                             that turn is stored.
  - storage failure AFTER  → the response is still returned to the user;
    a successful LLM call    storage is best-effort.
"""

import sqlite3

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

import memory.episodic_memory as episodic_module
from tests.conftest import TEST_API_KEY


TURN_1 = "my son Arjun visits on sundays"
TURN_2 = "please remind me to call Arjun"


def llm_error(kind):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if kind == "timeout":
        return anthropic.APITimeoutError(request=req)
    if kind == "connection":
        return anthropic.APIConnectionError(request=req)
    if kind == "overloaded":
        return anthropic.InternalServerError(
            "overloaded", response=httpx.Response(529, request=req),
            body=None)
    return RuntimeError("unexpected client bug")


def state(orch, agent="rajan"):
    mo = orch.memory_store.orchestrator
    return {
        "wm": orch.memory_store.get_context(agent, limit=50),
        "episodes": sorted(
            (ep.episode_id, ep.content, ep.review_count)
            for ep in mo.episodic_memory.get_all()),
        "db": sorted(
            (ep.episode_id, ep.content)
            for ep in episodic_module.EpisodicMemoryStore().get_all()),
        "qdrant": mo.semantic_memory.count(),
    }


def failing_client(orch, error):
    real = orch.client._create

    def create(**kwargs):
        orch.client.requests.append(kwargs)
        raise error
    orch.client.messages.create = create
    return lambda: setattr(orch.client.messages, "create", real)


# ─── LLM failure ───────────────────────────────────────────

class TestLlmFailure:

    @pytest.mark.parametrize(
        "kind", ["timeout", "connection", "overloaded", "bug"])
    def test_propagates_and_stores_nothing(self, real_orchestrator, kind):
        orch = real_orchestrator()
        orch.route(TURN_1, agent_id="rajan")
        before = state(orch)
        error = llm_error(kind)
        restore = failing_client(orch, error)

        with pytest.raises(type(error)):
            orch.route(TURN_2, agent_id="rajan")

        after = state(orch)
        assert after["wm"] == before["wm"]
        assert after["db"] == before["db"]
        assert after["qdrant"] == before["qdrant"]
        assert TURN_2 not in [c for _, c, _ in after["episodes"]]

        restore()
        orch.route("who visits on sundays?", agent_id="rajan")
        sent = [m["content"] for m in orch.client.last_messages]
        assert TURN_1 in sent
        assert TURN_2 not in sent

    def test_api_returns_generic_500(self, real_orchestrator, monkeypatch):
        import importlib
        import sys

        import dotenv
        import orchestrator.orchestrator as orch_module

        orch = real_orchestrator()
        failing_client(orch, llm_error("timeout"))
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
        monkeypatch.setenv("MAEM_API_KEY", TEST_API_KEY)
        monkeypatch.setattr(orch_module, "Orchestrator", lambda: orch)
        sys.modules.pop("api.main", None)
        app = importlib.import_module("api.main").app
        client = TestClient(app, raise_server_exceptions=False)

        r = client.post("/chat", json={"message": TURN_1, "user_id": "rajan"},
                        headers={"X-API-Key": TEST_API_KEY})
        sys.modules.pop("api.main", None)

        assert r.status_code == 500
        assert r.text == "Internal Server Error"  # no internals leaked
        assert orch.memory_store.get_context("rajan") == []

    # Phase 9: PlannerAgent now reinforces / records feedback only
    # after the LLM call succeeds (was a strict xfail).
    def test_failed_turn_records_no_positive_feedback(
        self, real_orchestrator
    ):
        orch = real_orchestrator()
        orch.route(TURN_1, agent_id="rajan")
        failing_client(orch, llm_error("timeout"))

        with pytest.raises(anthropic.APITimeoutError):
            orch.route("when does Arjun visit?", agent_id="rajan")

        for ep in orch.memory_store.orchestrator.episodic_memory.get_all():
            assert ep.agent_feedback == pytest.approx(0.5), ep.content

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Needs a decision: Orchestrator builds anthropic.Anthropic() "
            "with SDK defaults (600 s timeout, 2 retries), so one hung "
            "API call can hold a /chat request for many minutes."
        ),
    )
    def test_llm_client_has_bounded_timeout(self, real_orchestrator,
                                            monkeypatch):
        import orchestrator.orchestrator as orch_module
        seen = {}

        class Recorder:
            def __init__(self, *a, **k):
                seen.update(k)
                self.messages = None
        orch = real_orchestrator()
        monkeypatch.setattr(orch_module, "anthropic",
                            type("m", (), {"Anthropic": Recorder}))
        orch_module.Orchestrator().stop()
        assert "timeout" in seen and seen["timeout"] <= 120


# ─── Storage failure after a successful LLM response ───────

class TestStorageFailureAfterLlm:

    def break_semantic(self, orch, monkeypatch, only_role=None):
        sem = orch.memory_store.orchestrator.semantic_memory
        real_add = sem.add

        def add(memory_id, content, embedding, metadata=None):
            if only_role is None or (metadata or {}).get("role") == only_role:
                raise RuntimeError("Qdrant unavailable")
            return real_add(memory_id=memory_id, content=content,
                            embedding=embedding, metadata=metadata)
        monkeypatch.setattr(sem, "add", add)

    def test_response_still_returned(self, real_orchestrator, monkeypatch):
        orch = real_orchestrator()
        self.break_semantic(orch, monkeypatch)

        result = orch.route(TURN_1, agent_id="rajan")

        assert result["response"] == "Okay, noted."
        assert len(orch.client.requests) == 1

    def test_semantic_failure_rolls_back_episodic(
        self, real_orchestrator, monkeypatch
    ):
        orch = real_orchestrator()
        self.break_semantic(orch, monkeypatch)

        orch.route(TURN_1, agent_id="rajan")

        s = state(orch)
        assert s["episodes"] == [] and s["db"] == [] and s["qdrant"] == 0
        # documented: working memory is never rolled back
        assert [m["content"] for m in s["wm"]] == [TURN_1]

    def test_sqlite_failure_leaves_no_orphan(
        self, real_orchestrator, monkeypatch
    ):
        """
        Regression: EpisodicMemoryStore.add used to append to RAM before
        saving, so a failed save left an episode retrievable until the
        next restart even though store() raised.
        """
        orch = real_orchestrator()
        ep_store = orch.memory_store.orchestrator.episodic_memory

        def locked(ep):
            raise sqlite3.OperationalError("database is locked")
        monkeypatch.setattr(ep_store, "_save_episode", locked)

        result = orch.route(TURN_1, agent_id="rajan")

        assert result["response"] == "Okay, noted."
        assert ep_store.get_all() == []
        assert orch.memory_store.orchestrator.semantic_memory.count() == 0

    def test_reply_store_failure_keeps_user_turn(
        self, real_orchestrator, monkeypatch
    ):
        orch = real_orchestrator()
        self.break_semantic(orch, monkeypatch, only_role="assistant")

        result = orch.route(TURN_1, agent_id="rajan")

        assert result["response"] == "Okay, noted."
        s = state(orch)
        assert [c for _, c in s["db"]] == [TURN_1]
        assert s["qdrant"] == 1

    def test_next_turn_still_works(self, real_orchestrator, monkeypatch):
        orch = real_orchestrator()
        sem = orch.memory_store.orchestrator.semantic_memory
        self.break_semantic(orch, monkeypatch)
        orch.route(TURN_1, agent_id="rajan")
        del sem.add  # drop the instance override → real method again

        orch.route(TURN_2, agent_id="rajan")

        assert {"role": "user", "content": TURN_1} in orch.client.last_messages

    def test_response_embedding_failure_still_returns(
        self, real_orchestrator, monkeypatch
    ):
        """
        Characterisation: the reply is embedded inside the best-effort
        block BEFORE the user turn is stored, so an embedding failure
        there drops the whole turn from memory (all tiers) — but the
        user still gets the answer.
        """
        orch = real_orchestrator()
        calls = {"n": 0}
        real_embed = orch.executor.embed_fn

        def flaky(text):
            calls["n"] += 1
            if text == "Okay, noted.":
                raise RuntimeError("embedding model crashed")
            return real_embed(text)
        monkeypatch.setattr(orch.executor, "embed_fn", flaky)

        result = orch.route(TURN_1, agent_id="rajan")

        assert result["response"] == "Okay, noted."
        s = state(orch)
        assert s["wm"] == [] and s["db"] == [] and s["qdrant"] == 0

    def test_over_http_storage_failure_is_200(
        self, real_orchestrator, monkeypatch
    ):
        import importlib
        import sys

        import dotenv
        import orchestrator.orchestrator as orch_module

        orch = real_orchestrator()
        self.break_semantic(orch, monkeypatch)
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
        monkeypatch.setenv("MAEM_API_KEY", TEST_API_KEY)
        monkeypatch.setattr(orch_module, "Orchestrator", lambda: orch)
        sys.modules.pop("api.main", None)
        app = importlib.import_module("api.main").app

        r = TestClient(app).post(
            "/chat", json={"message": TURN_1, "user_id": "rajan"},
            headers={"X-API-Key": TEST_API_KEY})
        sys.modules.pop("api.main", None)

        assert r.status_code == 200
        assert r.json()["reply"] == "Okay, noted."


# ─── Retrieval failure (before the LLM) ────────────────────

class TestRetrievalFailure:

    def test_vector_store_outage_fails_request_before_llm(
        self, real_orchestrator, monkeypatch
    ):
        """
        Characterisation: retrieval is NOT best-effort. A Qdrant outage
        fails /chat outright and the LLM is never called.
        """
        orch = real_orchestrator()
        sem = orch.memory_store.orchestrator.semantic_memory

        def down(*a, **k):
            raise RuntimeError("Qdrant unavailable")
        monkeypatch.setattr(sem, "search", down)

        with pytest.raises(RuntimeError, match="Qdrant"):
            orch.route(TURN_1, agent_id="rajan")
        assert orch.client.requests == []
