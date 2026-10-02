"""
Phase 9 — retrieval quality & efficiency.

Covers the relevance cut-off, assistant-reply exclusion, forgotten
memories never resurfacing, relevance ranking, and the episodic
vector cache (no per-query re-embedding).
"""

from datetime import timedelta

import pytest

import memory.episodic_memory as episodic_module
from tests.conftest import hash_embed


def store(real_memory, content, agent_id="rajan", role="user"):
    return real_memory.store.store(
        agent_id=agent_id,
        content=content,
        embedding=hash_embed(content),
        context={"role": role},
    )["memory_id"]


def retrieve(real_memory, query, agent_id="rajan", **kw):
    return real_memory.store.retrieve(
        agent_id=agent_id, query=query, embedding=hash_embed(query), **kw
    )


# ─── Relevance cut-off ─────────────────────────────────────

class TestRelevanceCutoff:

    def test_unrelated_memory_not_returned_by_any_tier(self, real_memory):
        store(real_memory, "rajan favourite colour blue")
        r = retrieve(real_memory, "tomatoes garden")
        assert r["fused_memory"] == []
        assert r["semantic_memory"] == []
        assert r["episodic_memory"] == []

    def test_related_memory_returned(self, real_memory):
        mid = store(real_memory, "rajan grows tomatoes")
        r = retrieve(real_memory, "tomatoes garden")
        assert [m.episode_id for m in r["fused_memory"]] == [mid]
        assert r["scores"][mid] >= r["min_score"]

    def test_cutoff_is_configurable(self, real_memory, monkeypatch):
        store(real_memory, "rajan grows tomatoes")
        monkeypatch.setenv("RETRIEVAL_MIN_SCORE", "0.99")
        assert retrieve(real_memory, "tomatoes garden")["fused_memory"] == []
        monkeypatch.setenv("RETRIEVAL_MIN_SCORE", "0.0")
        assert retrieve(real_memory, "tomatoes garden")["fused_memory"]

    @pytest.mark.parametrize("raw", ["abc", "nan", ""])
    def test_bad_cutoff_falls_back_to_default(self, monkeypatch, raw):
        from memory import retrieval_config
        monkeypatch.setenv("RETRIEVAL_MIN_SCORE", raw)
        assert (
            retrieval_config.min_relevance_score()
            == retrieval_config.DEFAULT_MIN_SCORE
        )

    def test_semantic_scores_are_raw_cosine(self, real_memory):
        store(real_memory, "rajan grows tomatoes")
        r = retrieve(real_memory, "rajan grows tomatoes")
        assert r["semantic_memory"][0].similarity == pytest.approx(1.0)


# ─── Ranking ───────────────────────────────────────────────

class TestRanking:

    def test_most_relevant_first(self, real_memory):
        weak = store(real_memory, "rajan tomatoes")
        strong = store(real_memory, "rajan tomatoes garden soil")
        r = retrieve(real_memory, "tomatoes garden soil")
        ids = [m.episode_id for m in r["fused_memory"]]
        assert ids.index(strong) < ids.index(weak)


# ─── Assistant replies ─────────────────────────────────────

class TestAssistantReplies:

    def test_assistant_reply_not_retrieved(self, real_memory, monkeypatch):
        monkeypatch.delenv("RETRIEVE_ASSISTANT_REPLIES", raising=False)
        store(real_memory, "your tomatoes look great", role="assistant")
        assert retrieve(real_memory, "tomatoes great")["fused_memory"] == []

    def test_assistant_reply_retrieved_when_enabled(
        self, real_memory, monkeypatch
    ):
        monkeypatch.setenv("RETRIEVE_ASSISTANT_REPLIES", "true")
        store(real_memory, "your tomatoes look great", role="assistant")
        assert retrieve(real_memory, "tomatoes great")["fused_memory"]

    def test_assistant_reply_still_in_working_memory(self, real_memory):
        store(real_memory, "your tomatoes look great", role="assistant")
        assert real_memory.working.size("rajan") == 1


# ─── Forgotten memories ────────────────────────────────────

class TestForgotten:

    def test_forgotten_but_unpruned_memory_not_retrieved(self, real_memory):
        mid = store(real_memory, "rajan grows tomatoes")
        ep = real_memory.episodic.get_by_id(mid)
        ep.last_reviewed_at -= timedelta(hours=10_000)
        assert ep.is_forgotten()
        # still physically in the vector index (not pruned yet)
        assert real_memory.semantic.get(mid) is not None

        r = retrieve(real_memory, "tomatoes garden")
        assert r["fused_memory"] == []


# ─── Efficiency: episodic vector cache ─────────────────────

class TestVectorCache:

    def test_stored_memories_never_reembedded_on_query(
        self, real_memory, monkeypatch
    ):
        for i in range(25):
            store(real_memory, f"rajan fact number {i} about tomatoes")

        calls = []
        monkeypatch.setattr(
            episodic_module, "simple_embedding",
            lambda text: calls.append(text) or hash_embed(text),
        )
        for _ in range(5):
            retrieve(real_memory, "tomatoes facts")

        assert calls == []

    def test_reloaded_memories_embedded_once(self, real_memory, monkeypatch):
        for i in range(10):
            store(real_memory, f"rajan fact number {i} about tomatoes")

        calls = []
        monkeypatch.setattr(
            episodic_module, "simple_embedding",
            lambda text: calls.append(text) or hash_embed(text),
        )
        # fresh store = restart: loads from SQLite with no vectors cached
        reloaded = episodic_module.EpisodicMemoryStore()
        for _ in range(3):
            reloaded.grounded_retrieve_scored(
                query="tomatoes", embedding=hash_embed("tomatoes"),
                top_k=5, agent_id="rajan",
            )

        assert len(calls) == 10

    def test_removed_memory_drops_cached_vector(self, real_memory):
        mid = store(real_memory, "rajan grows tomatoes")
        assert mid in real_memory.episodic._vectors
        real_memory.episodic.delete(mid)
        assert mid not in real_memory.episodic._vectors

    def test_returns_scores_best_first(self, real_memory):
        store(real_memory, "rajan tomatoes")
        store(real_memory, "rajan tomatoes garden soil")
        scored = real_memory.episodic.grounded_retrieve_scored(
            query="q", embedding=hash_embed("tomatoes garden soil"),
            top_k=5, agent_id="rajan", min_score=0.0,
        )
        scores = [s for s, _ in scored]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, real_memory):
        for i in range(12):
            store(real_memory, f"rajan tomatoes note {i}")
        r = retrieve(real_memory, "tomatoes", top_k=3)
        assert len(r["episodic_memory"]) == 3
        assert len(r["semantic_memory"]) == 3


# ─── Benchmark smoke test ──────────────────────────────────

class TestRetrievalBenchmark:

    def test_phase9_beats_baseline_on_noise(self, real_memory):
        # real_memory points the episodic DB at a pytest tmp dir
        from evaluation import retrieval_eval

        # the benchmark's own embedder (no stopword list), as used by
        # evaluation/evaluate.py — numbers match RESULTS.md section 4
        rows, sweep, best_cutoff, reembeds = retrieval_eval.run_benchmark(
            retrieval_eval.hash_embed)
        base, new = rows[0][1], rows[1][1]

        assert new["noise_rejection"] == 1.0
        assert new["irrelevant_per_query"] < base["irrelevant_per_query"]
        assert new["precision"] > base["precision"]
        assert new["assistant_leaks"] == 0
        assert reembeds == 0

    def test_refuses_to_touch_a_non_temporary_db(self, monkeypatch):
        from evaluation import retrieval_eval

        monkeypatch.setattr(
            episodic_module, "DB_PATH", "/srv/real/multi_agent_memory.db")
        with pytest.raises(RuntimeError):
            retrieval_eval.build_store(hash_embed)
