# tests/test_edge_cases.py
"""
Boundaries and malformed input across the memory stack, on REAL objects.

The rule every "rejected" test checks: invalid input fails with a
ValueError (HTTP 400) BEFORE any tier is written — no half-stored memory
in working memory, RAM, SQLite or Qdrant. Several of these failed that
rule before this phase (see CHANGELOG_PHASE7.md).
"""

import json
import sqlite3
from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import memory.episodic_memory as episodic_module
from memory.working_memory import MAX_CONTENT_LENGTH, TRUNCATION_SUFFIX
from tests.conftest import hash_embed


# ─── helpers ───────────────────────────────────────────────

def db_rows(real_memory):
    conn = sqlite3.connect(real_memory.db_path)
    try:
        return conn.execute(
            "SELECT episode_id, agent_id, content, context FROM episodes"
        ).fetchall()
    finally:
        conn.close()


def assert_nothing_stored(real_memory):
    assert real_memory.working.metrics()["current_size"] == 0
    assert real_memory.episodic.get_all() == []
    assert db_rows(real_memory) == []
    assert real_memory.semantic.count() == 0


def store(real_memory, content="hello there", agent_id="rajan",
          embedding=None, context=None):
    return real_memory.store.store(
        agent_id=agent_id,
        content=content,
        embedding=hash_embed(content) if embedding is None else embedding,
        context=context,
    )


@pytest.fixture
def client(api_app):
    # raise_server_exceptions=False → see the status a real client gets
    return TestClient(api_app.app, raise_server_exceptions=False)


# ─── Empty content ─────────────────────────────────────────

class TestEmptyContent:

    @pytest.mark.parametrize("content", ["", "   ", "\n\t"])
    def test_rejected_nothing_written(self, real_memory, content):
        with pytest.raises(ValueError, match="content cannot be empty"):
            store(real_memory, content=content, embedding=hash_embed("x"))
        assert_nothing_stored(real_memory)

    def test_api_returns_400(self, client, auth_headers, real_memory):
        r = client.post("/store", headers=auth_headers, json={
            "agent_id": "rajan", "content": "  ",
            "embedding": hash_embed("x")})
        assert r.status_code == 400
        assert "content" in r.json()["detail"]
        assert_nothing_stored(real_memory)

    def test_empty_chat_message_is_400(self, client, auth_headers):
        r = client.post("/chat", headers=auth_headers,
                        json={"message": "   ", "user_id": "rajan"})
        assert r.status_code == 400


# ─── Long content / truncation ─────────────────────────────

class TestLongContent:

    @pytest.mark.parametrize("extra", [1, 500, 90_000])
    def test_working_memory_truncated_exactly(self, real_memory, extra):
        content = "a" * (MAX_CONTENT_LENGTH + extra)

        store(real_memory, content=content)

        [item] = real_memory.working.as_messages(agent_id="rajan")
        assert len(item["content"]) == MAX_CONTENT_LENGTH
        assert item["content"].endswith(TRUNCATION_SUFFIX)
        keep = MAX_CONTENT_LENGTH - len(TRUNCATION_SUFFIX)
        assert item["content"][:keep] == content[:keep]
        assert real_memory.working.metrics()["total_truncations"] == 1

    def test_exactly_at_limit_not_truncated(self, real_memory):
        content = "b" * MAX_CONTENT_LENGTH
        store(real_memory, content=content)
        [item] = real_memory.working.as_messages(agent_id="rajan")
        assert item["content"] == content

    def test_long_term_tiers_keep_full_content(self, real_memory):
        content = "word " * 5000  # 25 000 chars
        mid = store(real_memory, content=content)["memory_id"]

        assert real_memory.episodic.get_by_id(mid).content == content
        assert real_memory.semantic.get(mid).content == content
        [(_, _, on_disk, _)] = db_rows(real_memory)
        assert on_disk == content
        reloaded = episodic_module.EpisodicMemoryStore().get_by_id(mid)
        assert reloaded.content == content

    def test_multibyte_truncation_is_clean(self, real_memory):
        content = "राजन🙂" * 4000  # well past the limit, multibyte
        mid = store(real_memory, content=content)["memory_id"]

        [item] = real_memory.working.as_messages(agent_id="rajan")
        assert len(item["content"]) == MAX_CONTENT_LENGTH
        item["content"].encode("utf-8")  # no broken code points
        reloaded = episodic_module.EpisodicMemoryStore().get_by_id(mid)
        assert reloaded.content == content

    def test_llm_gets_full_task_but_truncated_history(
        self, real_orchestrator
    ):
        """Characterisation: history is truncated, the current task isn't."""
        orch = real_orchestrator()
        long_task = "note " * 3000

        orch.route(long_task, agent_id="rajan")
        assert orch.client.last_messages[-1]["content"] == long_task.strip()

        orch.route("next thing", agent_id="rajan")
        history_user = orch.client.last_messages[0]["content"]
        assert len(history_user) == MAX_CONTENT_LENGTH
        assert history_user.endswith(TRUNCATION_SUFFIX)

    def test_oversized_memory_is_skipped_not_blocking(self, real_memory):
        """
        Phase 9: _budgeted_fusion SKIPS a memory that can't fit in the
        remaining token budget instead of stopping. One memory longer
        than the whole budget (~4 000 words) no longer empties
        fused_memory — the small relevant memory still reaches the
        LLM. (Previously characterised in CHANGELOG_PHASE7.md.)
        """
        store(real_memory, content="rajan grows tomatoes in the garden")
        big = " ".join(["garden"] * 4000)
        store(real_memory, content=big)

        r = real_memory.store.retrieve(
            agent_id="rajan", query="garden", embedding=hash_embed("garden"))

        assert len(r["episodic_memory"]) == 2
        assert [m.content for m in r["fused_memory"]] == [
            "rajan grows tomatoes in the garden"
        ]

    def test_quotes_newlines_and_nul_roundtrip(self, real_memory):
        content = "it's \"quoted\"; DROP TABLE episodes;--\nline2\x00end"
        mid = store(real_memory, content=content)["memory_id"]
        assert episodic_module.EpisodicMemoryStore().get_by_id(
            mid).content == content
        assert len(db_rows(real_memory)) == 1


# ─── Embeddings ────────────────────────────────────────────

BAD_EMBEDDINGS = {
    "too_short": [0.1] * 383,
    "too_long": [0.1] * 385,
    "empty": [],
    "tuple": tuple([0.1] * 384),
    "strings": ["0.1"] * 384,
    "none_value": [None] * 384,
    "nan": [float("nan")] + [0.1] * 383,
    "inf": [float("inf")] + [0.1] * 383,
    "bools": [True] * 384,
}


class TestBadEmbeddings:

    @pytest.mark.parametrize("name", BAD_EMBEDDINGS)
    def test_store_rejected_before_any_write(self, real_memory, name):
        with pytest.raises(ValueError):
            store(real_memory, embedding=BAD_EMBEDDINGS[name])
        assert_nothing_stored(real_memory)

    @pytest.mark.parametrize("name", ["too_short", "nan", "bools"])
    def test_retrieve_rejected(self, real_memory, name):
        store(real_memory)
        with pytest.raises(ValueError):
            real_memory.store.retrieve(
                agent_id="rajan", query="q",
                embedding=BAD_EMBEDDINGS[name])

    @pytest.mark.parametrize("path", ["/store", "/retrieve"])
    def test_api_wrong_dimension_is_400(
        self, client, auth_headers, real_memory, path
    ):
        body = {"agent_id": "rajan", "embedding": [0.1] * 10}
        body.update({"content": "hi"} if path == "/store" else {"query": "q"})
        r = client.post(path, headers=auth_headers, json=body)
        assert r.status_code == 400
        assert "dimension" in r.json()["detail"]
        assert_nothing_stored(real_memory)

    def test_api_non_numeric_is_422(self, client, auth_headers):
        r = client.post("/store", headers=auth_headers, json={
            "agent_id": "rajan", "content": "hi",
            "embedding": ["x"] * 384})
        assert r.status_code == 422


# ─── Context dicts ─────────────────────────────────────────

class TestMalformedContext:

    @pytest.mark.parametrize("context", [
        ["role", "user"],
        "user",
        {"role": "admin"},
        {"role": None},
        {"shared": "no"},      # used to leave an orphan episode in RAM
        {"shared": "true"},
        {"shared": 1},
        {"confidence": "high"},
        {"confidence": float("nan")},
        {"when": datetime(2026, 1, 1)},  # used to corrupt DB context
        {"tags": {"a", "b"}},
    ])
    def test_rejected_before_any_write(self, real_memory, context):
        with pytest.raises(ValueError):
            store(real_memory, context=context)
        assert_nothing_stored(real_memory)

    @pytest.mark.parametrize("context", [
        None, {}, {"role": "assistant"}, {"shared": True},
        {"confidence": 0}, {"nested": {"a": [1, 2, {"b": None}]}},
        {"unknown_key": "kept"},
    ])
    def test_valid_contexts_roundtrip_through_sqlite(
        self, real_memory, context
    ):
        mid = store(real_memory, context=context)["memory_id"]

        reloaded = episodic_module.EpisodicMemoryStore().get_by_id(mid)
        assert isinstance(reloaded.context, dict)
        for key, value in (context or {}).items():
            assert reloaded.context[key] == value

    def test_api_bad_role_is_400(self, client, auth_headers, real_memory):
        r = client.post("/store", headers=auth_headers, json={
            "agent_id": "rajan", "content": "hi",
            "embedding": hash_embed("hi"), "context": {"role": "admin"}})
        assert r.status_code == 400
        assert_nothing_stored(real_memory)

    def test_api_non_dict_context_is_422(self, client, auth_headers):
        r = client.post("/store", headers=auth_headers, json={
            "agent_id": "rajan", "content": "hi",
            "embedding": hash_embed("hi"), "context": ["x"]})
        assert r.status_code == 422

    def test_legacy_non_dict_context_row_does_not_break_metrics(
        self, real_memory
    ):
        """
        A DB written before this fix may already hold a stringified
        context. Loading it must not crash metrics() for everyone.
        """
        mid = store(real_memory)["memory_id"]
        conn = sqlite3.connect(real_memory.db_path)
        conn.execute("UPDATE episodes SET context=? WHERE episode_id=?",
                     (json.dumps("{'when': datetime(...)}"), mid))
        conn.commit()
        conn.close()

        fresh = episodic_module.EpisodicMemoryStore()
        assert isinstance(fresh.get_by_id(mid).context, dict)
        real_memory.orchestrator.episodic_memory = fresh
        assert real_memory.orchestrator.metrics("rajan")["total_memories"] == 1


# ─── Unusual agent_ids ─────────────────────────────────────

ODD_IDS = [
    "rajan smith",
    "a/b",
    "../rajan",
    "राजन",
    "José 🙂",
    "rajan' OR '1'='1",
    "%",
    "*",
    "rajan\n",
    "x" * 500,
]


class TestUnusualAgentIds:

    @pytest.mark.parametrize("agent_id", ODD_IDS)
    def test_roundtrip_and_isolation(self, real_memory, agent_id):
        store(real_memory, content="rajan private diary", agent_id="rajan")
        mid = store(real_memory, content="odd agent private diary",
                    agent_id=agent_id)["memory_id"]
        q = hash_embed("private diary")

        own = real_memory.store.retrieve(agent_id=agent_id,
                                         query="private diary", embedding=q)
        rajan = real_memory.store.retrieve(agent_id="rajan",
                                           query="private diary", embedding=q)

        assert {m.content for m in own["fused_memory"]} == \
            {"odd agent private diary"}
        assert {m.content for m in rajan["fused_memory"]} == \
            {"rajan private diary"}
        assert [m["content"] for m in
                real_memory.store.get_context(agent_id)] == \
            ["odd agent private diary"]
        reloaded = episodic_module.EpisodicMemoryStore().get_by_id(mid)
        assert reloaded.agent_id == agent_id

    @pytest.mark.parametrize("agent_id", ["", "   "])
    def test_blank_agent_id_rejected(self, real_memory, agent_id):
        with pytest.raises(ValueError):
            store(real_memory, agent_id=agent_id)
        assert_nothing_stored(real_memory)

    def test_unicode_agent_id_over_http(self, client, auth_headers):
        client.post("/store", headers=auth_headers, json={
            "agent_id": "राजन", "content": "namaste diary",
            "embedding": hash_embed("namaste diary")})
        r = client.get("/context/राजन", headers=auth_headers)
        assert [m["content"] for m in r.json()] == ["namaste diary"]

    def test_slash_agent_id_writable_but_not_readable_over_http(
        self, client, auth_headers
    ):
        """
        Characterisation: an agent_id containing "/" can be written via
        POST /store (body), but no GET /{agent_id} route can address it —
        %2F is decoded before routing, so it's a 404.
        """
        r = client.post("/store", headers=auth_headers, json={
            "agent_id": "a/b", "content": "slashy",
            "embedding": hash_embed("slashy")})
        assert r.status_code == 200
        assert client.get("/context/a%2Fb",
                          headers=auth_headers).status_code == 404

    def test_none_agent_id_means_all_agents_in_python_api(self, real_memory):
        """
        Characterisation of a latent footgun: MemoryStore.retrieve(
        agent_id=None) disables scoping and returns every agent's data.
        Not reachable over HTTP (see test_security), but any future
        Python caller passing None gets cross-agent results.
        """
        store(real_memory, content="rajan diary", agent_id="rajan")
        store(real_memory, content="alice diary", agent_id="alice")

        r = real_memory.store.retrieve(agent_id=None, query="diary",
                                       embedding=hash_embed("diary"))
        assert {m.content for m in r["fused_memory"]} == \
            {"rajan diary", "alice diary"}


# ─── Nonexistent ids ───────────────────────────────────────

class TestNonexistentIds:

    @pytest.mark.parametrize("memory_id", [str(uuid4()), "not-a-uuid", ""])
    def test_reinforce_unknown_is_harmless(self, real_memory, memory_id):
        mid = store(real_memory)["memory_id"]
        before = real_memory.episodic.get_by_id(mid).review_count

        real_memory.store.reinforce(agent_id="rajan", memory_id=memory_id)

        assert real_memory.episodic.get_by_id(mid).review_count == before
        assert real_memory.semantic.get(mid).reinforcement_count == 0
        assert real_memory.semantic.count() == 1

    @pytest.mark.parametrize("memory_id", [str(uuid4()), "not-a-uuid"])
    def test_delete_unknown_is_harmless(self, real_memory, memory_id):
        store(real_memory)
        assert real_memory.episodic.delete(memory_id) is False
        real_memory.semantic.delete(memory_id)  # no raise
        assert len(real_memory.episodic.get_all()) == 1
        assert real_memory.semantic.count() == 1

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Needs a decision: POST /reinforce with an unknown memory_id "
            "answers 200 {'status': 'reinforced'} although nothing was "
            "reinforced. /feedback returns 404 in the same situation."
        ),
    )
    def test_api_reinforce_unknown_is_404(self, client, auth_headers):
        r = client.post("/reinforce", headers=auth_headers,
                        json={"agent_id": "rajan", "memory_id": str(uuid4())})
        assert r.status_code == 404

    # /feedback on forgotten / pruned / unknown memories is covered in
    # tests/test_feedback.py (TestMissingMemory, TestFeedbackEndpoint).

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_retrieve_non_positive_top_k_is_client_error(
        self, client, auth_headers, top_k
    ):
        r = client.post("/retrieve", headers=auth_headers, json={
            "agent_id": "rajan", "query": "q",
            "embedding": hash_embed("q"), "top_k": top_k})
        assert 400 <= r.status_code < 500
