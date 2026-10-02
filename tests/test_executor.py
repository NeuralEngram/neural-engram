# tests/test_executor.py

import pytest
from unittest.mock import MagicMock, patch, call
from agents.executor_agent import ExecutorAgent


# ─── Fixtures ──────────────────────────────────────────────

def make_embed_fn():
    """Returns a deterministic fake embedding."""
    return lambda text: [0.1, 0.2, 0.3]

def make_llm_fn(response="LLM response"):
    return lambda task, context: response

def make_memory_store():
    store = MagicMock()
    store.retrieve.return_value = {"fused_memory": []}
    return store

def make_agent(llm_response="LLM response"):
    memory_store = make_memory_store()
    agent = ExecutorAgent(
        memory_store=memory_store,
        embed_fn=make_embed_fn(),
        llm_fn=make_llm_fn(llm_response),
        agent_id="test_executor"
    )
    # stub base agent methods
    agent.retrieve_memories = MagicMock(return_value={"fused_memory": []})
    agent.get_context = MagicMock(return_value=[])
    agent.store_memory = MagicMock()
    return agent


# ─── Init ──────────────────────────────────────────────────

class TestExecutorAgentInit:

    def test_agent_id_set(self):
        agent = make_agent()
        assert agent.agent_id == "test_executor"

    def test_default_agent_id(self):
        store = make_memory_store()
        agent = ExecutorAgent(
            memory_store=store,
            embed_fn=make_embed_fn(),
            llm_fn=make_llm_fn()
        )
        agent.retrieve_memories = MagicMock(return_value={"fused_memory": []})
        agent.get_context = MagicMock(return_value=[])
        agent.store_memory = MagicMock()
        assert agent.agent_id == "executor_agent"

    def test_embed_fn_stored(self):
        agent = make_agent()
        assert callable(agent.embed_fn)

    def test_llm_fn_stored(self):
        agent = make_agent()
        assert callable(agent.llm_fn)


# ─── execute() input validation ───────────────────────────

class TestExecuteValidation:

    def test_empty_task_raises(self):
        agent = make_agent()
        with pytest.raises(ValueError, match="task cannot be empty"):
            agent.execute("")

    def test_whitespace_only_task_raises(self):
        agent = make_agent()
        with pytest.raises(ValueError, match="task cannot be empty"):
            agent.execute("   ")

    def test_task_is_stripped(self):
        agent = make_agent()
        result = agent.execute("  hello  ")
        assert result["task"] == "hello"


# ─── execute() return shape ────────────────────────────────

class TestExecuteReturnShape:

    def test_returns_all_keys(self):
        agent = make_agent()
        result = agent.execute("do something")
        assert set(result.keys()) == {
            "task", "response", "augmented_context", "retrieved_memories"
        }

    def test_task_in_result(self):
        agent = make_agent()
        result = agent.execute("run task")
        assert result["task"] == "run task"

    def test_response_in_result(self):
        agent = make_agent(llm_response="done")
        result = agent.execute("run task")
        assert result["response"] == "done"

    def test_augmented_context_is_list(self):
        agent = make_agent()
        result = agent.execute("task")
        assert isinstance(result["augmented_context"], list)

    def test_retrieved_memories_in_result(self):
        agent = make_agent()
        result = agent.execute("task")
        assert "fused_memory" in result["retrieved_memories"]


# ─── execute() default context ────────────────────────────

class TestExecuteDefaultContext:

    def test_default_context_is_user_role(self):
        agent = make_agent()
        agent.execute("task")
        calls = agent.store_memory.call_args_list
        # first store_memory call is for the user task
        first_call_context = calls[0].kwargs.get("context") or calls[0].args[2]
        assert first_call_context == {"role": "user"}

    def test_custom_context_passed_through(self):
        agent = make_agent()
        custom = {"role": "user", "session": "abc"}
        agent.execute("task", context=custom)
        calls = agent.store_memory.call_args_list
        first_call_context = calls[0].kwargs.get("context") or calls[0].args[2]
        assert first_call_context == custom


# ─── execute() memory flow ────────────────────────────────

class TestExecuteMemoryFlow:

    def test_retrieve_called_before_store(self):
        call_order = []
        agent = make_agent()
        agent.retrieve_memories = MagicMock(
            side_effect=lambda **kw: call_order.append("retrieve") or {"fused_memory": []}
        )
        agent.store_memory = MagicMock(
            side_effect=lambda **kw: call_order.append("store")
        )
        agent.execute("task")
        assert call_order[0] == "retrieve"
        assert "store" in call_order

    def test_embed_called_for_task(self):
        calls = []
        agent = make_agent()
        agent.embed_fn = lambda text: calls.append(text) or [0.1, 0.2, 0.3]
        agent.execute("my task")
        assert "my task" in calls

    def test_embed_called_for_response(self):
        calls = []
        agent = make_agent(llm_response="my response")
        agent.embed_fn = lambda text: calls.append(text) or [0.1, 0.2, 0.3]
        agent.execute("my task")
        assert "my response" in calls

    def test_store_memory_called_twice(self):
        agent = make_agent()
        agent.execute("task")
        assert agent.store_memory.call_count == 2

    def test_assistant_memory_stored(self):
        agent = make_agent(llm_response="assistant reply")
        agent.execute("task")
        calls = agent.store_memory.call_args_list
        contents = [
            c.kwargs.get("content") or c.args[0]
            for c in calls
        ]
        assert "assistant reply" in contents

    def test_retrieve_called_with_task_embedding(self):
        agent = make_agent()
        agent.execute("task")
        agent.retrieve_memories.assert_called_once()
        call_kwargs = agent.retrieve_memories.call_args.kwargs
        assert call_kwargs["query"] == "task"
        assert call_kwargs["embedding"] == [0.1, 0.2, 0.3]


# ─── _build_memory_context ────────────────────────────────

class TestBuildMemoryContext:

    def test_empty_fused_returns_empty(self):
        agent = make_agent()
        result = agent._build_memory_context({"fused_memory": []})
        assert result == []

    def test_missing_fused_key_returns_empty(self):
        agent = make_agent()
        result = agent._build_memory_context({})
        assert result == []

    def test_single_memory_builds_context(self):
        agent = make_agent()
        mem = MagicMock()
        mem.content = "I love Python"
        result = agent._build_memory_context({"fused_memory": [mem]})
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert "I love Python" in result[0]["content"]

    def test_multiple_memories_in_single_block(self):
        agent = make_agent()
        m1, m2 = MagicMock(), MagicMock()
        m1.content = "fact one"
        m2.content = "fact two"
        result = agent._build_memory_context({"fused_memory": [m1, m2]})
        assert len(result) == 1
        assert "fact one" in result[0]["content"]
        assert "fact two" in result[0]["content"]

    def test_context_block_has_system_role(self):
        agent = make_agent()
        mem = MagicMock()
        mem.content = "something"
        result = agent._build_memory_context({"fused_memory": [mem]})
        assert result[0]["role"] == "system"


# ─── context merging ──────────────────────────────────────

class TestContextMerging:

    def test_system_messages_merged(self):
        agent = make_agent()
        agent.get_context = MagicMock(return_value=[
            {"role": "system", "content": "base system prompt"}
        ])
        mem = MagicMock()
        mem.content = "retrieved memory"
        agent.retrieve_memories = MagicMock(return_value={"fused_memory": [mem]})
        result = agent.execute("task")
        system_msgs = [
            m for m in result["augmented_context"]
            if m["role"] == "system"
        ]
        assert len(system_msgs) == 1
        assert "base system prompt" in system_msgs[0]["content"]
        assert "retrieved memory" in system_msgs[0]["content"]

    def test_non_system_messages_preserved(self):
        agent = make_agent()
        agent.get_context = MagicMock(return_value=[
            {"role": "user", "content": "previous user message"},
            {"role": "assistant", "content": "previous reply"},
        ])
        result = agent.execute("task")
        roles = [m["role"] for m in result["augmented_context"]]
        assert "user" in roles
        assert "assistant" in roles

    def test_no_memories_no_extra_system_block(self):
        agent = make_agent()
        agent.get_context = MagicMock(return_value=[])
        agent.retrieve_memories = MagicMock(return_value={"fused_memory": []})
        result = agent.execute("task")
        system_msgs = [
            m for m in result["augmented_context"]
            if m["role"] == "system"
        ]
        assert len(system_msgs) == 0


# ─── LLM error handling ───────────────────────────────────

class TestLLMErrorHandling:

    def test_llm_exception_propagates(self):
        agent = make_agent()
        agent.llm_fn = MagicMock(side_effect=RuntimeError("LLM down"))
        with pytest.raises(RuntimeError, match="LLM down"):
            agent.execute("task")

    def test_storage_failure_does_not_suppress_response(self):
        agent = make_agent(llm_response="valid response")
        agent.store_memory = MagicMock(side_effect=Exception("DB error"))
        result = agent.execute("task")
        assert result["response"] == "valid response"


# ─── respond() ────────────────────────────────────────────

class TestRespond:

    def test_returns_string(self):
        agent = make_agent(llm_response="hello")
        assert agent.respond("say hi") == "hello"

    def test_respond_calls_execute(self):
        agent = make_agent(llm_response="ok")
        agent.execute = MagicMock(return_value={
            "task": "ping",
            "response": "pong",
            "augmented_context": [],
            "retrieved_memories": {}
        })
        result = agent.respond("ping")
        agent.execute.assert_called_once_with(task="ping")
        assert result == "pong"

    def test_respond_no_context_injection(self):
        """respond() should not expose context parameter."""
        import inspect
        sig = inspect.signature(ExecutorAgent.respond)
        assert "context" not in sig.parameters

# ═══════════════════════════════════════════════════════════
# REAL memory objects (no mocks)
# ═══════════════════════════════════════════════════════════
#
# make_agent() above replaces get_context / retrieve_memories /
# store_memory with MagicMocks, so no test could notice execute()
# reading the *executor's own* working-memory bucket instead of the
# conversation's (debug-pass bug #2). These run the real BaseAgent
# methods against a real MemoryStore.

class _RecordingLLM:
    def __init__(self):
        self.contexts = []

    def __call__(self, task, context):
        self.contexts.append([dict(m) for m in context])
        return "Okay, noted."


class TestExecutorOnRealMemory:

    @pytest.fixture
    def agent_and_llm(self, real_memory, embed):
        llm = _RecordingLLM()
        agent = ExecutorAgent(
            memory_store=real_memory.store, embed_fn=embed, llm_fn=llm
        )
        return agent, llm

    def test_second_turn_gets_first_turn_as_context(
        self, agent_and_llm, real_memory
    ):
        agent, llm = agent_and_llm

        agent.execute("my son is called Arjun", agent_id="rajan")
        agent.execute("who is Arjun?", agent_id="rajan")

        convo = [m for m in llm.contexts[1] if m["role"] != "system"]
        assert convo == [
            {"role": "user", "content": "my son is called Arjun"},
            {"role": "assistant", "content": "Okay, noted."},
        ]

    def test_writes_to_conversation_bucket_not_own(
        self, agent_and_llm, real_memory
    ):
        agent, _ = agent_and_llm

        agent.execute("my son is called Arjun", agent_id="rajan")

        assert real_memory.working.size("rajan") == 2
        assert real_memory.working.is_empty(agent.agent_id)
        assert real_memory.working.is_empty("default")

    def test_retrieved_memory_reaches_llm_system_prompt(
        self, agent_and_llm, real_memory, embed
    ):
        agent, llm = agent_and_llm
        real_memory.store.store(
            agent_id="rajan", content="rajan son Arjun lives in Pune",
            embedding=embed("rajan son Arjun lives in Pune"),
        )

        agent.execute("where does Arjun live?", agent_id="rajan")

        system = [m for m in llm.contexts[0] if m["role"] == "system"]
        assert len(system) == 1
        assert system[0]["content"].count(
            "rajan son Arjun lives in Pune") == 1

    def test_other_agents_memory_never_reaches_llm(
        self, agent_and_llm, real_memory
    ):
        agent, llm = agent_and_llm

        agent.execute("alice pin is 9921", agent_id="alice")
        agent.execute("what is the pin?", agent_id="bob")

        flat = " ".join(m["content"] for m in llm.contexts[1])
        assert "9921" not in flat

    def test_stored_episodes_belong_to_conversation(
        self, agent_and_llm, real_memory
    ):
        agent, _ = agent_and_llm

        agent.execute("my son is called Arjun", agent_id="rajan")

        roles = sorted(
            (ep.agent_id, ep.context.get("role"))
            for ep in real_memory.episodic.get_all()
        )
        assert roles == [("rajan", "assistant"), ("rajan", "user")]
