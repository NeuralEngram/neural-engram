# tests/test_episodic_memory.py

import math
import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest

from memory.episodic_memory import (
    Episode,
    RecallStatus,
    compute_importance,
    should_store_memory,
    cosine_similarity,
    safe_json_dumps,
    safe_json_loads,
    simple_embedding,
    _heuristic_importance,
    get_connection,
    W_ACCESS_FREQ, W_TASK_IMPORTANCE,
    W_AGENT_FEEDBACK, W_TASK_SUCCESS, W_CONSENSUS,
    W_ACCESS_FREQ_P, W_TASK_IMPORTANCE_P,
    W_AGENT_FEEDBACK_P, W_TASK_SUCCESS_P,
    _importance_cache, _importance_cache_lock,
    MAX_IMPORTANCE_CACHE,
)
from memory.ebbinghaus import DEFAULT_STABILITY_HOURS


# ─── Fixtures ──────────────────────────────────────────────

def make_episode(**kwargs) -> Episode:
    defaults = dict(
        content="test content",
        context={"key": "value"},
        agent_id="agent_1",
        tags=["test"],
        shared=False,
        stability_hours=DEFAULT_STABILITY_HOURS,
        importance=0.6,
        agent_feedback=0.5,
        task_success_rate=0.5,
        review_count=0,
        cross_agent_recall_count=0,
    )
    defaults.update(kwargs)
    return Episode(**defaults)


NOW = datetime.now(timezone.utc)


# ─── Weight Invariants ─────────────────────────────────────

class TestWeightInvariants:

    def test_shared_weights_sum_to_one(self):
        total = (
            W_ACCESS_FREQ + W_TASK_IMPORTANCE +
            W_AGENT_FEEDBACK + W_TASK_SUCCESS + W_CONSENSUS
        )
        assert abs(total - 1.0) < 1e-9

    def test_private_weights_sum_to_one(self):
        total = (
            W_ACCESS_FREQ_P + W_TASK_IMPORTANCE_P +
            W_AGENT_FEEDBACK_P + W_TASK_SUCCESS_P
        )
        assert abs(total - 1.0) < 1e-6

    def test_consensus_weight_positive(self):
        assert 0.0 < W_CONSENSUS < 1.0


# ─── safe_json_dumps / safe_json_loads ────────────────────

class TestJsonHelpers:

    def test_dumps_dict(self):
        result = safe_json_dumps({"a": 1})
        assert result == '{"a": 1}'

    def test_dumps_non_serializable(self):
        obj = object()
        result = safe_json_dumps(obj)
        assert isinstance(result, str)

    def test_loads_valid_json(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_loads_invalid_json(self):
        assert safe_json_loads("not json") == "not json"

    def test_loads_none(self):
        result = safe_json_loads(None)
        assert result is None

    def test_roundtrip(self):
        data = {"x": [1, 2, 3], "y": "hello"}
        assert safe_json_loads(safe_json_dumps(data)) == data


# ─── cosine_similarity ────────────────────────────────────

class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_partial_similarity(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = cosine_similarity(a, b)
        assert 0.0 < sim < 1.0


# ─── _heuristic_importance ────────────────────────────────

class TestHeuristicImportance:

    def test_high_keywords(self):
        for phrase in ["I am a developer", "my goal is X", "I need help"]:
            assert _heuristic_importance(phrase) == pytest.approx(0.75)

    def test_low_keywords(self):
        for phrase in ["okay", "thanks", "hello", "hi there"]:
            assert _heuristic_importance(phrase) == pytest.approx(0.15)

    def test_neutral_text(self):
        score = _heuristic_importance("The weather is nice today")
        assert score == pytest.approx(0.45)

    def test_case_insensitive(self):
        assert _heuristic_importance("I AM here") == pytest.approx(0.75)
        assert _heuristic_importance("THANKS") == pytest.approx(0.15)


# ─── should_store_memory ──────────────────────────────────

class TestShouldStoreMemory:

    def test_above_threshold_stores(self):
        assert should_store_memory("hello world this is a test", 0.5) is True

    def test_below_threshold_rejects(self):
        assert should_store_memory("some text here", 0.10) is False

    def test_ultra_short_high_importance_stores(self):
        assert should_store_memory("ok", 0.25) is True

    def test_ultra_short_low_importance_rejects(self):
        assert should_store_memory("ok", 0.16) is False

    def test_exact_threshold(self):
    # 3 words, so ultra-short guard doesn't apply
        assert should_store_memory("hello world today", 0.15) is True

    def test_ultra_short_exact_boundary(self):
            assert should_store_memory("hi", 0.20) is True
            assert should_store_memory("hi", 0.19) is False


# ─── compute_importance ───────────────────────────────────

class TestComputeImportance:

    def setup_method(self):
        # clear cache before each test
        with _importance_cache_lock:
            _importance_cache.clear()

    def test_short_text_returns_heuristic(self):
        score = compute_importance("hi", "agent_1")
        assert score == pytest.approx(0.2)

    def test_question_returns_heuristic(self):
        score = compute_importance("What is the capital of France?", "agent_1")
        assert score == pytest.approx(0.15)

    def test_result_cached(self):
        compute_importance("hi", "agent_1")
        with _importance_cache_lock:
            assert len(_importance_cache) == 1
        # second call should hit cache
        compute_importance("hi", "agent_1")
        with _importance_cache_lock:
            assert len(_importance_cache) == 1

    def test_different_agents_different_cache_keys(self):
        compute_importance("hi", "agent_1")
        compute_importance("hi", "agent_2")
        with _importance_cache_lock:
            assert len(_importance_cache) == 2

    def test_llm_fallback_on_exception(self):
        with patch("memory.episodic_memory.get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API down")
            score = compute_importance("I love building things with code", "agent_1")
            assert 0.0 <= score <= 1.0

    def test_llm_score_clamped(self):
        mock_resp = MagicMock()
        mock_resp.content[0].text = "1.5"
        with patch("memory.episodic_memory.get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_resp
            score = compute_importance("I love building things with code", "agent_1")
            assert score <= 1.0

    def test_cache_eviction_when_full(self):
        with _importance_cache_lock:
            for i in range(MAX_IMPORTANCE_CACHE):
                _importance_cache[f"key_{i}"] = 0.5

        # this should trigger eviction
        compute_importance("hi there world", "agent_evict")
        with _importance_cache_lock:
            assert len(_importance_cache) <= MAX_IMPORTANCE_CACHE


# ─── Episode dataclass ────────────────────────────────────

class TestEpisodeDefaults:

    def test_episode_id_unique(self):
        e1 = make_episode()
        e2 = make_episode()
        assert e1.episode_id != e2.episode_id

    def test_created_at_is_utc(self):
        e = make_episode()
        assert e.created_at.tzinfo is not None

    def test_default_tags_empty(self):
        e = Episode(content="x", context={}, agent_id="a")
        assert e.tags == []

    def test_default_shared_false(self):
        e = Episode(content="x", context={}, agent_id="a")
        assert e.shared is False


# ─── Episode.retention ────────────────────────────────────

class TestEpisodeRetention:

    def test_fresh_retention_is_one(self):
        e = make_episode(last_reviewed_at=NOW)
        assert e.retention(now=NOW) == pytest.approx(1.0)

    def test_retention_decays_over_time(self):
        past = NOW - timedelta(hours=DEFAULT_STABILITY_HOURS)
        e = make_episode(last_reviewed_at=past)
        r = e.retention(now=NOW)
        assert r == pytest.approx(math.exp(-1), rel=1e-4)

    def test_retention_between_zero_and_one(self):
        past = NOW - timedelta(hours=100)
        e = make_episode(last_reviewed_at=past)
        r = e.retention(now=NOW)
        assert 0.0 <= r <= 1.0


# ─── Episode.is_forgotten ─────────────────────────────────

class TestEpisodeIsForgotten:

    def test_fresh_not_forgotten(self):
        e = make_episode(last_reviewed_at=NOW)
        assert e.is_forgotten(now=NOW) is False

    def test_old_memory_forgotten(self):
        very_old = NOW - timedelta(hours=100_000)
        e = make_episode(last_reviewed_at=very_old)
        assert e.is_forgotten(now=NOW) is True


# ─── Episode.access_frequency_score ──────────────────────

class TestAccessFrequencyScore:

    def test_zero_reviews(self):
        e = make_episode(review_count=0)
        assert e.access_frequency_score() == pytest.approx(0.0)

    def test_increases_with_reviews(self):
        e1 = make_episode(review_count=1)
        e2 = make_episode(review_count=10)
        assert e2.access_frequency_score() > e1.access_frequency_score()

    def test_capped_at_one(self):
        e = make_episode(review_count=10_000)
        assert e.access_frequency_score() <= 1.0

    def test_hundred_reviews_is_one(self):
        e = make_episode(review_count=100)
        assert e.access_frequency_score() == pytest.approx(1.0)


# ─── Episode.consensus_score ──────────────────────────────

class TestConsensusScore:

    def test_private_memory_zero(self):
        e = make_episode(shared=False, cross_agent_recall_count=10)
        assert e.consensus_score() == pytest.approx(0.0)

    def test_shared_zero_recalls(self):
        e = make_episode(shared=True, cross_agent_recall_count=0)
        assert e.consensus_score() == pytest.approx(0.0)

    def test_shared_increases_with_recalls(self):
        e1 = make_episode(shared=True, cross_agent_recall_count=1)
        e2 = make_episode(shared=True, cross_agent_recall_count=10)
        assert e2.consensus_score() > e1.consensus_score()

    def test_capped_at_one(self):
        e = make_episode(shared=True, cross_agent_recall_count=10_000)
        assert e.consensus_score() <= 1.0

    def test_twenty_recalls_is_one(self):
        e = make_episode(shared=True, cross_agent_recall_count=20)
        assert e.consensus_score() == pytest.approx(1.0)


# ─── Episode.priority_score ───────────────────────────────

class TestPriorityScore:

    def test_private_score_between_zero_and_one(self):
        e = make_episode(shared=False, last_reviewed_at=NOW)
        score = e.priority_score(now=NOW)
        assert 0.0 <= score <= 1.0

    def test_shared_score_between_zero_and_one(self):
        e = make_episode(shared=True, last_reviewed_at=NOW)
        score = e.priority_score(now=NOW)
        assert 0.0 <= score <= 1.0

    def test_forgotten_memory_low_priority(self):
        very_old = NOW - timedelta(hours=100_000)
        e = make_episode(last_reviewed_at=very_old)
        score = e.priority_score(now=NOW)
        assert score < 0.01

    def test_higher_importance_higher_score(self):
        e_low = make_episode(importance=0.1, last_reviewed_at=NOW)
        e_high = make_episode(importance=0.9, last_reviewed_at=NOW)
        assert e_high.priority_score(now=NOW) > e_low.priority_score(now=NOW)

    def test_shared_vs_private_with_consensus(self):
        e_private = make_episode(shared=False, last_reviewed_at=NOW)
        e_shared = make_episode(
            shared=True, cross_agent_recall_count=20, last_reviewed_at=NOW
        )
        assert e_shared.priority_score(now=NOW) > e_private.priority_score(now=NOW)

    def test_clamped_feedback_and_success(self):
        # out-of-range values should be clamped, not crash
        e = make_episode(
            agent_feedback=2.0,
            task_success_rate=-1.0,
            last_reviewed_at=NOW
        )
        score = e.priority_score(now=NOW)
        assert 0.0 <= score <= 1.0

    def test_retention_acts_as_multiplier(self):
        e_fresh = make_episode(last_reviewed_at=NOW)
        e_old = make_episode(
            last_reviewed_at=NOW - timedelta(hours=DEFAULT_STABILITY_HOURS * 3)
        )
        assert e_fresh.priority_score(now=NOW) > e_old.priority_score(now=NOW)


# ─── Thread-local DB connections ──────────────────────────

class TestGetConnection:

    def test_returns_connection(self, tmp_path):
        db = str(tmp_path / "test.db")
        conn = get_connection(db)
        assert conn is not None

    def test_same_thread_same_connection(self, tmp_path):
        db = str(tmp_path / "test.db")
        c1 = get_connection(db)
        c2 = get_connection(db)
        assert c1 is c2

    def test_different_threads_different_connections(self, tmp_path):
        db = str(tmp_path / "test.db")
        connections = []

        def grab():
            connections.append(get_connection(db))

        t1, t2 = threading.Thread(target=grab), threading.Thread(target=grab)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert connections[0] is not connections[1]


# ─── RecallStatus ─────────────────────────────────────────

class TestRecallStatus:

    def test_values(self):
        assert RecallStatus.UPDATED.value == "updated"
        assert RecallStatus.FORGOTTEN.value == "forgotten"
        assert RecallStatus.NOT_FOUND.value == "not_found"