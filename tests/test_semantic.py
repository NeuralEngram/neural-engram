"""
Tests for SemanticMemoryStore (semantic_memory.py)

Run with:
    pip install pytest qdrant-client --break-system-packages
    pytest test_semantic_memory.py -v
"""

import math
import threading
import uuid
import pytest

from memory.semantic_memory import (
    SemanticMemory,
    SemanticMemoryStore,
    VECTOR_SIZE,
    SIMILARITY_WEIGHT,
    REINFORCEMENT_WEIGHT,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def uid() -> str:
    """Return a fresh valid UUID string (required by Qdrant local storage)."""
    return str(uuid.uuid4())


def make_embedding(value: float = 0.1, size: int = VECTOR_SIZE) -> list:
    """Return a normalised-ish embedding of fixed size."""
    return [value] * size


def make_store(tmp_path) -> SemanticMemoryStore:
    """Return a store backed by a temporary directory."""
    return SemanticMemoryStore(storage_path=str(tmp_path))


# ─────────────────────────────────────────────────────────────
# SemanticMemory dataclass
# ─────────────────────────────────────────────────────────────

class TestSemanticMemoryDataclass:

    def test_frozen_prevents_mutation(self):
        mem = SemanticMemory(memory_id=uid(), content="hello", metadata={})
        with pytest.raises((AttributeError, TypeError)):
            mem.content = "changed"  # type: ignore[misc]

    def test_defaults(self):
        mem = SemanticMemory(memory_id=uid(), content="x", metadata={})
        assert mem.score is None
        assert mem.embedding is None
        assert mem.reinforcement_count == 0

    def test_fields_stored(self):
        mid = uid()
        mem = SemanticMemory(
            memory_id=mid,
            content="test content",
            metadata={"key": "val"},
            score=0.95,
            reinforcement_count=3,
        )
        assert mem.memory_id == mid
        assert mem.content == "test content"
        assert mem.metadata == {"key": "val"}
        assert mem.score == 0.95
        assert mem.reinforcement_count == 3


# ─────────────────────────────────────────────────────────────
# Embedding validation
# ─────────────────────────────────────────────────────────────

class TestValidateEmbedding:

    @pytest.fixture
    def store(self, tmp_path):
        return make_store(tmp_path)

    def test_valid_embedding_passes(self, store):
        store._validate_embedding(make_embedding())  # no exception

    def test_empty_embedding_raises(self, store):
        with pytest.raises(ValueError, match="empty"):
            store._validate_embedding([])

    def test_wrong_size_raises(self, store):
        with pytest.raises(ValueError, match=str(VECTOR_SIZE)):
            store._validate_embedding([0.1] * (VECTOR_SIZE - 1))

    def test_non_numeric_value_raises(self, store):
        bad = make_embedding()
        bad[0] = "not-a-number"  # type: ignore[list-item]
        with pytest.raises(ValueError, match="numeric"):
            store._validate_embedding(bad)

    def test_integer_values_accepted(self, store):
        store._validate_embedding([1] * VECTOR_SIZE)  # no exception


# ─────────────────────────────────────────────────────────────
# Reinforcement scoring
# ─────────────────────────────────────────────────────────────

class TestReinforcementScore:

    @pytest.fixture
    def store(self, tmp_path):
        return make_store(tmp_path)

    def test_zero_count_returns_zero(self, store):
        assert store._reinforcement_score(0) == 0.0

    def test_score_increases_with_count(self, store):
        s1 = store._reinforcement_score(1)
        s5 = store._reinforcement_score(5)
        s20 = store._reinforcement_score(20)
        assert 0 < s1 < s5 < s20

    def test_score_capped_at_one(self, store):
        assert store._reinforcement_score(10_000) <= 1.0

    def test_logarithmic_shape(self, store):
        for n in (1, 5, 10, 50):
            expected = min(math.log1p(n) / 3.0, 1.0)
            assert store._reinforcement_score(n) == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────
# Store initialisation
# ─────────────────────────────────────────────────────────────

class TestStoreInit:

    def test_creates_store(self, tmp_path):
        store = make_store(tmp_path)
        assert store.collection_name == "semantic_memory"
        assert store.vector_size == VECTOR_SIZE

    def test_custom_collection_name(self, tmp_path):
        store = SemanticMemoryStore(
            collection_name="my_collection",
            storage_path=str(tmp_path),
        )
        assert store.collection_name == "my_collection"

    def test_metrics_start_at_zero(self, tmp_path):
        store = make_store(tmp_path)
        m = store.metrics()
        assert m["total_searches"] == 0
        assert m["total_inserts"] == 0
        assert m["total_deletes"] == 0
        assert m["total_reinforcements"] == 0
        assert m["total_failures"] == 0

    def test_second_init_reuses_collection(self, tmp_path):
        # Should not raise even if collection already exists.
        make_store(tmp_path)
        make_store(tmp_path)  # no exception


# ─────────────────────────────────────────────────────────────
# add()
# ─────────────────────────────────────────────────────────────

class TestAdd:

    @pytest.fixture
    def store(self, tmp_path):
        return make_store(tmp_path)

    def test_returns_memory_id(self, store):
        mid = uid()
        result = store.add(mid, "hello world", make_embedding())
        assert result == mid

    def test_increments_insert_counter(self, store):
        store.add(uid(), "content", make_embedding())
        assert store.metrics()["total_inserts"] == 1

    def test_empty_content_raises(self, store):
        with pytest.raises(ValueError):
            store.add(uid(), "   ", make_embedding())

    def test_bad_embedding_raises(self, store):
        with pytest.raises(ValueError):
            store.add(uid(), "content", [])

    def test_count_increases(self, store):
        assert store.count() == 0
        store.add(uid(), "content", make_embedding())
        assert store.count() == 1

    def test_upsert_overwrites(self, store):
        mid = uid()
        store.add(mid, "first", make_embedding(0.1))
        store.add(mid, "second", make_embedding(0.2))
        mem = store.get(mid)
        assert mem.content == "second"

    def test_metadata_stored(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding(), metadata={"agent_id": "agent1"})
        mem = store.get(mid)
        assert mem.metadata.get("agent_id") == "agent1"


# ─────────────────────────────────────────────────────────────
# get()
# ─────────────────────────────────────────────────────────────

class TestGet:

    @pytest.fixture
    def store(self, tmp_path):
        return make_store(tmp_path)

    def test_returns_none_for_missing(self, store):
        assert store.get(uid()) is None

    def test_returns_memory_object(self, store):
        mid = uid()
        store.add(mid, "get me", make_embedding())
        mem = store.get(mid)
        assert isinstance(mem, SemanticMemory)
        assert mem.content == "get me"
        assert mem.memory_id == mid

    def test_with_vectors_false_hides_embedding(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding())
        mem = store.get(mid, with_vectors=False)
        assert mem.embedding is None

    def test_with_vectors_true_returns_embedding(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding())
        mem = store.get(mid, with_vectors=True)
        assert mem.embedding is not None
        assert len(mem.embedding) == VECTOR_SIZE


# ─────────────────────────────────────────────────────────────
# search()
# ─────────────────────────────────────────────────────────────

class TestSearch:

    @pytest.fixture
    def store(self, tmp_path):
        return make_store(tmp_path)

    def _populate(self, store, n=3):
        for i in range(n):
            store.add(uid(), f"memory {i}", make_embedding(i * 0.01 + 0.01))

    def test_returns_list(self, store):
        self._populate(store)
        results = store.search(make_embedding())
        assert isinstance(results, list)

    def test_respects_limit(self, store):
        self._populate(store, n=5)
        results = store.search(make_embedding(), limit=2)
        assert len(results) <= 2

    def test_results_sorted_by_score_desc(self, store):
        self._populate(store)
        results = store.search(make_embedding())
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_composite(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding())
        results = store.search(make_embedding())
        mem = next(r for r in results if r.memory_id == mid)
        assert 0.0 < mem.score <= 1.0

    def test_metadata_filter_narrows_results(self, store):
        store.add(uid(), "agent A memory", make_embedding(0.1), metadata={"agent_id": "A"})
        store.add(uid(), "agent B memory", make_embedding(0.1), metadata={"agent_id": "B"})
        results = store.search(make_embedding(), metadata_filter={"agent_id": "A"})
        assert all(r.metadata.get("agent_id") == "A" for r in results)

    def test_increments_search_counter(self, store):
        self._populate(store)
        store.search(make_embedding())
        assert store.metrics()["total_searches"] == 1

    def test_invalid_embedding_raises(self, store):
        with pytest.raises(ValueError):
            store.search([])

    def test_empty_store_returns_empty_list(self, store):
        results = store.search(make_embedding())
        assert results == []


# ─────────────────────────────────────────────────────────────
# search_multi()
# ─────────────────────────────────────────────────────────────

class TestSearchMulti:

    @pytest.fixture
    def store(self, tmp_path):
        return make_store(tmp_path)

    def test_no_agent_id_returns_all(self, store):
        id_a = uid()
        id_b = uid()
        store.add(id_a, "content", make_embedding(), metadata={"agent_id": "A"})
        store.add(id_b, "content", make_embedding(), metadata={"agent_id": "B"})
        results = store.search_multi(make_embedding(), agent_id=None)
        ids = {r.memory_id for r in results}
        assert {id_a, id_b}.issubset(ids)

    def test_agent_id_filters(self, store):
        store.add(uid(), "content", make_embedding(), metadata={"agent_id": "A"})
        store.add(uid(), "content", make_embedding(), metadata={"agent_id": "B"})
        results = store.search_multi(make_embedding(), agent_id="A")
        assert all(r.metadata.get("agent_id") == "A" for r in results)


# ─────────────────────────────────────────────────────────────
# reinforce()
# ─────────────────────────────────────────────────────────────

class TestReinforce:

    @pytest.fixture
    def store(self, tmp_path):
        return make_store(tmp_path)

    def test_increments_reinforcement_count(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding())
        store.reinforce(mid)
        mem = store.get(mid)
        assert mem.reinforcement_count == 1

    def test_multiple_reinforcements_accumulate(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding())
        for _ in range(5):
            store.reinforce(mid)
        mem = store.get(mid)
        assert mem.reinforcement_count == 5

    def test_reinforcement_boosts_search_score(self, store):
        boosted = uid()
        plain = uid()
        store.add(boosted, "boosted", make_embedding())
        store.add(plain, "plain", make_embedding())
        for _ in range(10):
            store.reinforce(boosted)
        results = store.search(make_embedding())
        order = [r.memory_id for r in results]
        assert order.index(boosted) < order.index(plain)

    def test_reinforce_missing_does_not_raise(self, store):
        store.reinforce(uid())  # logs warning, no exception

    def test_increments_reinforcement_counter(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding())
        store.reinforce(mid)
        assert store.metrics()["total_reinforcements"] == 1


# ─────────────────────────────────────────────────────────────
# delete()
# ─────────────────────────────────────────────────────────────

class TestDelete:

    @pytest.fixture
    def store(self, tmp_path):
        return make_store(tmp_path)

    def test_removes_memory(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding())
        store.delete(mid)
        assert store.get(mid) is None

    def test_decrements_count(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding())
        assert store.count() == 1
        store.delete(mid)
        assert store.count() == 0

    def test_increments_delete_counter(self, store):
        mid = uid()
        store.add(mid, "content", make_embedding())
        store.delete(mid)
        assert store.metrics()["total_deletes"] == 1

    def test_delete_nonexistent_does_not_raise(self, store):
        store.delete(uid())  # Qdrant silently ignores missing IDs


# ─────────────────────────────────────────────────────────────
# count() and metrics()
# ─────────────────────────────────────────────────────────────

class TestCountAndMetrics:

    @pytest.fixture
    def store(self, tmp_path):
        return make_store(tmp_path)

    def test_count_zero_initially(self, store):
        assert store.count() == 0

    def test_count_reflects_inserts(self, store):
        for _ in range(4):
            store.add(uid(), "x", make_embedding())
        assert store.count() == 4

    def test_metrics_average_latency(self, store):
        store.add(uid(), "x", make_embedding())
        store.search(make_embedding())
        m = store.metrics()
        assert m["average_search_latency"] > 0.0

    def test_metrics_total_memories(self, store):
        store.add(uid(), "x", make_embedding())
        store.add(uid(), "y", make_embedding())
        m = store.metrics()
        assert m["total_memories"] == 2


# ─────────────────────────────────────────────────────────────
# Thread safety
# ─────────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_adds(self, tmp_path):
        store = make_store(tmp_path)
        errors = []

        def add_memory(i):
            try:
                store.add(uid(), f"content {i}", make_embedding(i * 0.001 + 0.001))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_memory, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent adds: {errors}"
        assert store.count() == 20

    def test_concurrent_reads_and_writes(self, tmp_path):
        store = make_store(tmp_path)
        base_id = uid()
        store.add(base_id, "content", make_embedding())
        errors = []

        def read():
            try:
                store.search(make_embedding(), limit=1)
            except Exception as e:
                errors.append(e)

        def write():
            try:
                store.add(uid(), "content", make_embedding())
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=read) for _ in range(10)]
            + [threading.Thread(target=write) for _ in range(10)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent read/write: {errors}"


# ─────────────────────────────────────────────────────────────
# Score formula
# ─────────────────────────────────────────────────────────────

class TestScoreFormula:

    def test_weights_sum_to_one(self):
        assert SIMILARITY_WEIGHT + REINFORCEMENT_WEIGHT == pytest.approx(1.0)

    def test_composite_score_formula(self, tmp_path):
        store = make_store(tmp_path)
        mid = uid()
        store.add(mid, "content", make_embedding())
        results = store.search(make_embedding())
        assert results, "Expected at least one result"
        mem = results[0]
        # reinforcement_count is 0 → reinforcement_score = 0
        # final_score = similarity * 0.8 + 0 * 0.2 = similarity * 0.8
        # score must be in [0, 1]
        assert 0.0 <= mem.score <= 1.0