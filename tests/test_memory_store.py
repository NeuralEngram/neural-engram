# tests/test_memory_store.py

import pytest
from unittest.mock import MagicMock
from memory.memory_store import MemoryStore


# ─── Fixtures ──────────────────────────────────────────────

def make_orchestrator():
    orc = MagicMock()
    orc.store.return_value = {"memory_id": "abc-123"}
    orc.retrieve.return_value = {"fused_memory": []}
    orc.get_context.return_value = []
    orc.reinforce.return_value = None
    orc.cleanup.return_value = 0
    orc.apply_decay.return_value = None
    orc.metrics.return_value = {"count": 5}
    return orc

def make_store():
    orc = make_orchestrator()
    return MemoryStore(orchestrator=orc), orc

EMBEDDING = [0.1, 0.2, 0.3]


# ─── Init ──────────────────────────────────────────────────

class TestMemoryStoreInit:

    def test_orchestrator_assigned(self):
        orc = make_orchestrator()
        store = MemoryStore(orchestrator=orc)
        assert store.orchestrator is orc


# ─── store() ───────────────────────────────────────────────

class TestStore:

    def test_returns_memory_id(self):
        store, _ = make_store()
        result = store.store(
            agent_id="agent_1",
            content="hello world",
            embedding=EMBEDDING
        )
        assert result == {"memory_id": "abc-123"}

    def test_delegates_to_orchestrator(self):
        store, orc = make_store()
        store.store(
            agent_id="agent_1",
            content="hello world",
            embedding=EMBEDDING,
            context={"role": "user"}
        )
        orc.store.assert_called_once_with(
            agent_id="agent_1",
            content="hello world",
            context={"role": "user"},
            embedding=EMBEDDING
        )

    def test_empty_agent_id_raises(self):
        store, _ = make_store()
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            store.store(
                agent_id="",
                content="hello",
                embedding=EMBEDDING
            )

    def test_whitespace_agent_id_raises(self):
        store, _ = make_store()
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            store.store(
                agent_id="   ",
                content="hello",
                embedding=EMBEDDING
            )

    def test_empty_content_raises(self):
        store, _ = make_store()
        with pytest.raises(ValueError, match="content cannot be empty"):
            store.store(
                agent_id="agent_1",
                content="",
                embedding=EMBEDDING
            )

    def test_whitespace_content_raises(self):
        store, _ = make_store()
        with pytest.raises(ValueError, match="content cannot be empty"):
            store.store(
                agent_id="agent_1",
                content="   ",
                embedding=EMBEDDING
            )

    def test_default_context_is_none(self):
        store, orc = make_store()
        store.store(
            agent_id="agent_1",
            content="hello",
            embedding=EMBEDDING
        )
        orc.store.assert_called_once_with(
            agent_id="agent_1",
            content="hello",
            context=None,
            embedding=EMBEDDING
        )


# ─── retrieve() ────────────────────────────────────────────

class TestRetrieve:

    def test_delegates_to_orchestrator(self):
        store, orc = make_store()
        store.retrieve(
            agent_id="agent_1",
            query="what do I know?",
            embedding=EMBEDDING,
            top_k=3,
            token_budget=2000
        )
        orc.retrieve.assert_called_once_with(
            agent_id="agent_1",
            query="what do I know?",
            embedding=EMBEDDING,
            top_k=3,
            token_budget=2000
        )

    def test_returns_dict(self):
        store, _ = make_store()
        result = store.retrieve(
            agent_id="agent_1",
            query="q",
            embedding=EMBEDDING
        )
        assert isinstance(result, dict)

    def test_default_top_k_is_five(self):
        store, orc = make_store()
        store.retrieve(
            agent_id="agent_1",
            query="q",
            embedding=EMBEDDING
        )
        _, kwargs = orc.retrieve.call_args
        assert kwargs["top_k"] == 5

    def test_default_token_budget_is_4000(self):
        store, orc = make_store()
        store.retrieve(
            agent_id="agent_1",
            query="q",
            embedding=EMBEDDING
        )
        _, kwargs = orc.retrieve.call_args
        assert kwargs["token_budget"] == 4000


# ─── get_context() ─────────────────────────────────────────

class TestGetContext:

    def test_delegates_to_orchestrator(self):
        store, orc = make_store()
        store.get_context(agent_id="agent_1", limit=5)
        orc.get_context.assert_called_once_with(
            agent_id="agent_1",
            limit=5
        )

    def test_returns_list(self):
        store, _ = make_store()
        result = store.get_context(agent_id="agent_1")
        assert isinstance(result, list)

    def test_default_limit_is_ten(self):
        store, orc = make_store()
        store.get_context(agent_id="agent_1")
        _, kwargs = orc.get_context.call_args
        assert kwargs["limit"] == 10


# ─── reinforce() ───────────────────────────────────────────

class TestReinforce:

    def test_delegates_to_orchestrator(self):
        store, orc = make_store()
        store.reinforce(agent_id="agent_1", memory_id="mem-001")
        orc.reinforce.assert_called_once_with(
            agent_id="agent_1",
            memory_id="mem-001"
        )

    def test_returns_none(self):
        store, _ = make_store()
        result = store.reinforce(
            agent_id="agent_1",
            memory_id="mem-001"
        )
        assert result is None


# ─── cleanup() ─────────────────────────────────────────────

class TestCleanup:

    def test_delegates_to_orchestrator(self):
        store, orc = make_store()
        store.cleanup()
        orc.cleanup.assert_called_once()

    def test_returns_int(self):
        store, _ = make_store()
        result = store.cleanup()
        assert isinstance(result, int)

    def test_returns_removed_count(self):
        store, orc = make_store()
        orc.cleanup.return_value = 7
        assert store.cleanup() == 7

    def test_zero_when_nothing_removed(self):
        store, orc = make_store()
        orc.cleanup.return_value = 0
        assert store.cleanup() == 0


# ─── apply_decay() ─────────────────────────────────────────

class TestApplyDecay:

    def test_delegates_to_orchestrator(self):
        store, orc = make_store()
        store.apply_decay()
        orc.apply_decay.assert_called_once()

    def test_returns_none(self):
        store, _ = make_store()
        assert store.apply_decay() is None


# ─── metrics() ─────────────────────────────────────────────

class TestMetrics:

    def test_returns_dict_when_orchestrator_has_metrics(self):
        store, _ = make_store()
        result = store.metrics()
        assert isinstance(result, dict)
        assert result == {"count": 5}

    def test_returns_empty_dict_when_no_metrics(self):
        orc = MagicMock(spec=[])  # no attributes at all
        store = MemoryStore(orchestrator=orc)
        result = store.metrics()
        assert result == {}

    def test_delegates_to_orchestrator_metrics(self):
        store, orc = make_store()
        store.metrics()
        orc.metrics.assert_called_once()