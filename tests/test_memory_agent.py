# tests/test_memory_agent.py

import pytest
from unittest.mock import MagicMock, call, patch
from agents.memory_agent import MemoryAgent


# ─── Fixtures ──────────────────────────────────────────────

def make_store():
    store = MagicMock()
    store.retrieve.return_value = {"fused_memory": []}
    return store

def make_agent(agent_id="memory_agent"):
    store = make_store()
    agent = MemoryAgent(memory_store=store, agent_id=agent_id)
    return agent, store

EMBEDDING = [0.1, 0.2, 0.3]


# ─── Init ──────────────────────────────────────────────────

class TestMemoryAgentInit:

    def test_default_agent_id(self):
        agent, _ = make_agent()
        assert agent.agent_id == "memory_agent"

    def test_custom_agent_id(self):
        agent, _ = make_agent(agent_id="tenant_1")
        assert agent.agent_id == "tenant_1"

    def test_memory_store_assigned(self):
        store = make_store()
        agent = MemoryAgent(memory_store=store)
        assert agent.memory_store is store


# ─── store_memory ──────────────────────────────────────────

class TestStoreMemory:

    def setup_method(self):
        self.agent, _ = make_agent()
        self.agent.memory_store = MagicMock()
        # NOTE: this used to assign
        #   BaseAgent.store_memory = MagicMock(...)
        # directly on the class with no teardown, which permanently
        # replaced store_memory for EVERY agent in the test process —
        # any later test using a real ExecutorAgent/PlannerAgent
        # silently stored nothing (the executor treats storage as
        # best-effort and swallows the error). Each test below already
        # patches with a self-restoring patch.object, so it's removed.

    def test_returns_memory_ids(self):
        with patch.object(
            self.agent.__class__.__bases__[0],
            "store_memory",
            return_value={"memory_id": "abc-123"}
        ):
            result = self.agent.store_memory(
                content="hello",
                embedding=EMBEDDING,
                context={"role": "user"}
            )
            assert result == {"memory_id": "abc-123"}

    def test_delegates_to_base(self):
        with patch.object(
            self.agent.__class__.__bases__[0],
            "store_memory",
            return_value={"memory_id": "xyz"}
        ) as mock_base:
            self.agent.store_memory(
                content="test",
                embedding=EMBEDDING,
                context={"role": "user"}
            )
            mock_base.assert_called_once_with(
                content="test",
                embedding=EMBEDDING,
                context={"role": "user"}
            )


# ─── retrieve_memories ─────────────────────────────────────

class TestRetrieveMemories:

    def test_delegates_to_base(self):
        agent, _ = make_agent()
        expected = {"fused_memory": [], "episodic": [], "semantic": []}
        with patch.object(
            agent.__class__.__bases__[0],
            "retrieve_memories",
            return_value=expected
        ) as mock_base:
            result = agent.retrieve_memories(
                query="what do I know?",
                embedding=EMBEDDING,
                top_k=3
            )
            mock_base.assert_called_once_with(
                query="what do I know?",
                embedding=EMBEDDING,
                top_k=3
            )
            assert result == expected

    def test_default_top_k_is_five(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "retrieve_memories",
            return_value={}
        ) as mock_base:
            agent.retrieve_memories(query="q", embedding=EMBEDDING)
            _, kwargs = mock_base.call_args
            assert kwargs.get("top_k", 5) == 5

    def test_returns_dict(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "retrieve_memories",
            return_value={"fused_memory": []}
        ):
            result = agent.retrieve_memories(query="q", embedding=EMBEDDING)
            assert isinstance(result, dict)


# ─── reinforce_memory ──────────────────────────────────────

class TestReinforceMemory:

    def test_delegates_to_base(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "reinforce_memory"
        ) as mock_base:
            agent.reinforce_memory(memory_id="mem-001")
            mock_base.assert_called_once_with(memory_id="mem-001")

    def test_returns_none(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "reinforce_memory",
            return_value=None
        ):
            result = agent.reinforce_memory(memory_id="mem-001")
            assert result is None


# ─── get_context ───────────────────────────────────────────

class TestGetContext:

    def test_delegates_to_base(self):
        agent, _ = make_agent()
        expected = [{"role": "user", "content": "hi"}]
        with patch.object(
            agent.__class__.__bases__[0],
            "get_context",
            return_value=expected
        ) as mock_base:
            result = agent.get_context(limit=5)
            mock_base.assert_called_once_with(limit=5)
            assert result == expected

    def test_default_limit_is_ten(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "get_context",
            return_value=[]
        ) as mock_base:
            agent.get_context()
            _, kwargs = mock_base.call_args
            assert kwargs.get("limit", 10) == 10

    def test_returns_list(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "get_context",
            return_value=[]
        ):
            result = agent.get_context()
            assert isinstance(result, list)


# ─── apply_decay ───────────────────────────────────────────

class TestApplyDecay:

    def test_delegates_to_base(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "apply_decay"
        ) as mock_base:
            agent.apply_decay()
            mock_base.assert_called_once()

    def test_returns_none(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "apply_decay",
            return_value=None
        ):
            assert agent.apply_decay() is None


# ─── cleanup_memories ──────────────────────────────────────

class TestCleanupMemories:

    def test_delegates_to_base(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "cleanup_memories",
            return_value=5
        ) as mock_base:
            result = agent.cleanup_memories()
            mock_base.assert_called_once()
            assert result == 5

    def test_returns_int(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "cleanup_memories",
            return_value=0
        ):
            result = agent.cleanup_memories()
            assert isinstance(result, int)

    def test_zero_when_nothing_removed(self):
        agent, _ = make_agent()
        with patch.object(
            agent.__class__.__bases__[0],
            "cleanup_memories",
            return_value=0
        ):
            assert agent.cleanup_memories() == 0


# ─── remember() ────────────────────────────────────────────

class TestRemember:

    def test_default_role_is_user(self):
        agent, _ = make_agent()
        agent.store_memory = MagicMock(return_value={"memory_id": "1"})
        agent.remember(content="hello", embedding=EMBEDDING)
        call_kwargs = agent.store_memory.call_args.kwargs
        assert call_kwargs["context"]["role"] == "user"

    def test_custom_role(self):
        agent, _ = make_agent()
        agent.store_memory = MagicMock(return_value={"memory_id": "1"})
        agent.remember(content="hello", embedding=EMBEDDING, role="assistant")
        call_kwargs = agent.store_memory.call_args.kwargs
        assert call_kwargs["context"]["role"] == "assistant"

    def test_extra_context_merged(self):
        agent, _ = make_agent()
        agent.store_memory = MagicMock(return_value={"memory_id": "1"})
        agent.remember(
            content="hello",
            embedding=EMBEDDING,
            extra_context={"session": "abc", "priority": 1}
        )
        call_kwargs = agent.store_memory.call_args.kwargs
        assert call_kwargs["context"]["session"] == "abc"
        assert call_kwargs["context"]["priority"] == 1
        assert call_kwargs["context"]["role"] == "user"

    def test_extra_context_none_no_crash(self):
        agent, _ = make_agent()
        agent.store_memory = MagicMock(return_value={"memory_id": "1"})
        agent.remember(content="hello", embedding=EMBEDDING, extra_context=None)
        agent.store_memory.assert_called_once()

    def test_extra_context_empty_dict_no_merge(self):
        agent, _ = make_agent()
        agent.store_memory = MagicMock(return_value={"memory_id": "1"})
        agent.remember(content="hello", embedding=EMBEDDING, extra_context={})
        call_kwargs = agent.store_memory.call_args.kwargs
        assert call_kwargs["context"] == {"role": "user"}

    def test_returns_memory_ids(self):
        agent, _ = make_agent()
        agent.store_memory = MagicMock(return_value={"memory_id": "xyz"})
        result = agent.remember(content="hello", embedding=EMBEDDING)
        assert result == {"memory_id": "xyz"}

    def test_mutable_default_not_shared(self):
        """extra_context=None default avoids mutable default argument bug."""
        agent, _ = make_agent()
        agent.store_memory = MagicMock(return_value={})
        agent.remember("a", EMBEDDING)
        agent.remember("b", EMBEDDING)
        # both calls should get independent context dicts
        calls = agent.store_memory.call_args_list
        ctx1 = calls[0].kwargs["context"]
        ctx2 = calls[1].kwargs["context"]
        assert ctx1 is not ctx2


# ─── recall() ──────────────────────────────────────────────

class TestRecall:

    def test_delegates_to_retrieve_memories(self):
        agent, _ = make_agent()
        agent.retrieve_memories = MagicMock(return_value={"fused_memory": []})
        agent.recall(query="what?", embedding=EMBEDDING, top_k=3)
        agent.retrieve_memories.assert_called_once_with(
            query="what?",
            embedding=EMBEDDING,
            top_k=3
        )

    def test_default_top_k_is_five(self):
        agent, _ = make_agent()
        agent.retrieve_memories = MagicMock(return_value={})
        agent.recall(query="q", embedding=EMBEDDING)
        _, kwargs = agent.retrieve_memories.call_args
        assert kwargs.get("top_k", 5) == 5

    def test_returns_dict(self):
        agent, _ = make_agent()
        agent.retrieve_memories = MagicMock(return_value={"fused_memory": []})
        result = agent.recall(query="q", embedding=EMBEDDING)
        assert isinstance(result, dict)


# ─── think() ───────────────────────────────────────────────

class TestThink:

    def test_delegates_to_retrieve_memories(self):
        agent, _ = make_agent()
        agent.retrieve_memories = MagicMock(return_value={"fused_memory": []})
        agent.think(query="deep thought", embedding=EMBEDDING, top_k=4)
        agent.retrieve_memories.assert_called_once_with(
            query="deep thought",
            embedding=EMBEDDING,
            top_k=4
        )

    def test_returns_full_results(self):
        agent, _ = make_agent()
        expected = {"fused_memory": [], "episodic": [], "semantic": []}
        agent.retrieve_memories = MagicMock(return_value=expected)
        result = agent.think(query="q", embedding=EMBEDDING)
        assert result == expected

    def test_default_top_k_is_five(self):
        agent, _ = make_agent()
        agent.retrieve_memories = MagicMock(return_value={"fused_memory": []})
        agent.think(query="q", embedding=EMBEDDING)
        _, kwargs = agent.retrieve_memories.call_args
        assert kwargs.get("top_k", 5) == 5

    def test_missing_fused_key_handled(self):
        agent, _ = make_agent()
        agent.retrieve_memories = MagicMock(return_value={})
        result = agent.think(query="q", embedding=EMBEDDING)
        assert isinstance(result, dict)

    def test_think_vs_recall_same_retrieval(self):
        """Both think() and recall() use retrieve_memories — same data."""
        agent, _ = make_agent()
        data = {"fused_memory": ["mem1"]}
        agent.retrieve_memories = MagicMock(return_value=data)
        assert agent.think("q", EMBEDDING) == agent.recall("q", EMBEDDING)