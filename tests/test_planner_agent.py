# tests/test_planner_agent.py
"""
PlannerAgent tests.

Unit tests for intent detection use no memory at all. Everything that
touches memory (reinforcement loop, plan() end-to-end) runs against the
REAL stack from conftest.real_memory — a real MemoryStore over real
WorkingMemory / EpisodicMemoryStore / SemanticMemoryStore — with a real
ExecutorAgent. Only the LLM is faked (it records what it was shown).

Several tests here fail against the pre-debug-pass code; each says which
bug it guards in its docstring.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents.executor_agent import ExecutorAgent
from agents.planner_agent import (
    GENERAL_SIGNALS,
    MEMORY_SIGNALS,
    REASONING_SIGNALS,
    PlannerAgent,
    TaskIntent,
)
from memory.episodic_memory import Episode
from memory.semantic_memory import SemanticMemory


# ─── Helpers ───────────────────────────────────────────────

class RecordingLLM:
    """Fake LLM: records every (task, context) it receives."""

    def __init__(self, response="Okay, noted."):
        self.response = response
        self.calls = []

    def __call__(self, task, context):
        self.calls.append((task, [dict(m) for m in context]))
        return self.response

    @property
    def last_context(self):
        return self.calls[-1][1]


def make_real_planner(real_memory, embed, llm=None):
    llm = llm or RecordingLLM()
    executor = ExecutorAgent(
        memory_store=real_memory.store,
        embed_fn=embed,
        llm_fn=llm,
    )
    planner = PlannerAgent(
        memory_store=real_memory.store,
        executor=executor,
        embed_fn=embed,
    )
    return planner, llm


def make_bare_planner():
    """Planner with no real memory — only for pure intent logic."""
    return PlannerAgent(
        memory_store=MagicMock(),
        executor=MagicMock(),
        embed_fn=lambda t: [0.0],
    )


def seed(real_memory, embed, agent_id, content, **context):
    """Store directly through the real MemoryStore; return memory_id."""
    return real_memory.store.store(
        agent_id=agent_id,
        content=content,
        embedding=embed(content),
        context=context or None,
    )["memory_id"]


def fused_id(mem):
    return getattr(mem, "episode_id", None) or getattr(mem, "memory_id")


# ─── Init ──────────────────────────────────────────────────

class TestInit:

    def test_default_agent_id(self):
        assert make_bare_planner().agent_id == "planner_agent"

    def test_empty_agent_id_rejected(self):
        with pytest.raises(ValueError):
            PlannerAgent(
                memory_store=MagicMock(),
                executor=MagicMock(),
                embed_fn=lambda t: [0.0],
                agent_id="   ",
            )


# ─── _detect_intent() ──────────────────────────────────────

class TestDetectIntent:

    @pytest.mark.parametrize("signal", MEMORY_SIGNALS)
    def test_each_memory_signal(self, signal):
        planner = make_bare_planner()
        assert planner._detect_intent(f"ok {signal} it") == TaskIntent.MEMORY

    @pytest.mark.parametrize("signal", REASONING_SIGNALS)
    def test_each_reasoning_signal(self, signal):
        planner = make_bare_planner()
        assert (
            planner._detect_intent(f"ok {signal} it")
            == TaskIntent.REASONING
        )

    @pytest.mark.parametrize("signal", GENERAL_SIGNALS)
    def test_each_general_signal(self, signal):
        planner = make_bare_planner()
        assert (
            planner._detect_intent(f"ok {signal} it")
            == TaskIntent.GENERAL
        )

    def test_no_signal_defaults_to_general(self):
        planner = make_bare_planner()
        assert planner._detect_intent("zzz qqq") == TaskIntent.GENERAL

    def test_case_insensitive(self):
        planner = make_bare_planner()
        assert planner._detect_intent("REMEMBER THIS") == TaskIntent.MEMORY
        assert planner._detect_intent("Analyze It") == TaskIntent.REASONING

    def test_memory_beats_reasoning(self):
        planner = make_bare_planner()
        task = "compare what we discussed last time"
        assert planner._detect_intent(task) == TaskIntent.MEMORY

    def test_memory_beats_general(self):
        planner = make_bare_planner()
        assert (
            planner._detect_intent("what is the thing you said")
            == TaskIntent.MEMORY
        )

    def test_reasoning_beats_general(self):
        planner = make_bare_planner()
        assert (
            planner._detect_intent("what is the difference between a and b")
            == TaskIntent.REASONING
        )

    @pytest.mark.parametrize("task", [
        "what is my daughter's name?",
        "where did I park",
        "where do I live",
        "is this mine",
    ])
    def test_personal_reference_is_memory(self, task):
        assert make_bare_planner()._detect_intent(task) == TaskIntent.MEMORY

    @pytest.mark.parametrize("task", [
        "hi", "Hello!", "ok thanks", "thank you so much", "good night",
    ])
    def test_small_talk(self, task):
        assert (
            make_bare_planner()._detect_intent(task)
            == TaskIntent.SMALL_TALK
        )

    @pytest.mark.parametrize("task", [
        "hi, how do I bake bread", "ok list it", "thanks, define gravity",
    ])
    def test_small_talk_words_mixed_with_content_are_not_small_talk(
        self, task
    ):
        assert (
            make_bare_planner()._detect_intent(task)
            != TaskIntent.SMALL_TALK
        )

    def test_matching_is_substring_based(self):
        """
        Characterisation, not an endorsement: signals match as raw
        substrings, so "reasonable" trips the "reason" signal. If you
        move to word-boundary matching, update this test on purpose.
        """
        planner = make_bare_planner()
        assert (
            planner._detect_intent("is this price reasonable")
            == TaskIntent.REASONING
        )


# ─── _requires_memory() ────────────────────────────────────

class TestRequiresMemory:

    @pytest.mark.parametrize(
        "intent",
        [TaskIntent.MEMORY, TaskIntent.REASONING, TaskIntent.GENERAL],
    )
    def test_always_true(self, intent):
        assert make_bare_planner()._requires_memory(intent) is True

    def test_general_task_still_pre_retrieves(self, real_memory, embed):
        planner, _ = make_real_planner(real_memory, embed)
        seed(real_memory, embed, "alice", "alice grows tomatoes")

        result = planner.plan("define tomatoes", agent_id="alice")

        assert result["intent"] == TaskIntent.GENERAL
        assert result["needs_memory"] is True
        assert result["planner_memories"]["fused_memory"]

    def test_small_talk_does_not_require_memory(self):
        assert make_bare_planner()._requires_memory(
            TaskIntent.SMALL_TALK) is False

    def test_false_skips_retrieval_entirely(self, real_memory, embed):
        """
        Phase 9 contract: one retrieval per turn, done by the planner.
        When memory isn't required nothing is retrieved anywhere and
        nothing is reinforced.
        """
        planner, llm = make_real_planner(real_memory, embed)
        mem_id = seed(real_memory, embed, "alice", "alice grows tomatoes")
        planner._requires_memory = lambda intent: False

        result = planner.plan("tell me about tomatoes", agent_id="alice")

        assert result["needs_memory"] is False
        assert result["planner_memories"] == {}
        assert result["retrieved_memories"]["fused_memory"] == []
        assert real_memory.episodic.get_by_id(mem_id).review_count == 0

    def test_one_retrieval_per_turn(self, real_memory, embed):
        planner, _ = make_real_planner(real_memory, embed)
        seed(real_memory, embed, "alice", "alice grows tomatoes")

        with patch.object(
            real_memory.store, "retrieve",
            wraps=real_memory.store.retrieve,
        ) as spy:
            planner.plan("tell me about tomatoes", agent_id="alice")

        assert spy.call_count == 1

    def test_task_embedded_once_per_turn(self, real_memory, embed):
        calls = []

        def counting_embed(text):
            calls.append(text)
            return embed(text)

        planner, _ = make_real_planner(real_memory, counting_embed)
        planner.plan("alice grows tomatoes", agent_id="alice")

        assert calls.count("alice grows tomatoes") == 1

    def test_small_talk_skips_retrieval(self, real_memory, embed):
        planner, llm = make_real_planner(real_memory, embed)
        seed(real_memory, embed, "alice", "hello alice grows tomatoes")

        with patch.object(
            real_memory.store, "retrieve",
            wraps=real_memory.store.retrieve,
        ) as spy:
            result = planner.plan("ok thanks!", agent_id="alice")

        assert result["intent"] == TaskIntent.SMALL_TALK
        assert spy.call_count == 0
        assert result["retrieved_memories"]["fused_memory"] == []

    @pytest.mark.parametrize("intent, expected", [
        (TaskIntent.MEMORY, 8),
        (TaskIntent.REASONING, 6),
        (TaskIntent.GENERAL, 4),
    ])
    def test_top_k_follows_intent(self, real_memory, embed, intent, expected):
        planner, _ = make_real_planner(real_memory, embed)
        planner._detect_intent = lambda task: intent

        with patch.object(
            real_memory.store, "retrieve",
            wraps=real_memory.store.retrieve,
        ) as spy:
            result = planner.plan("anything at all", agent_id="alice")

        assert spy.call_args.kwargs["top_k"] == expected
        assert result["top_k"] == expected


# ─── Fused-memory reinforcement loop ───────────────────────

class TestReinforcementLoop:
    """
    Setup used throughout:
      A = "alice planted tomatoes garden" — relevant; returned by the
          episodic tier, so it comes back as an Episode.
      B = "alice watered tomatoes daily"  — relevant (shares
          "tomatoes"); the episodic tier is made to miss it, so only
          the semantic tier returns it, as a SemanticMemory — the path
          debug-pass bug #3 broke.
      C = "alice favourite colour blue"   — shares no words with the
          query: below the relevance cut-off, so it is neither
          retrieved nor reinforced (Phase 9).
    """

    QUERY = "how is the tomatoes garden?"

    @pytest.fixture
    def seeded(self, real_memory, embed, monkeypatch):
        a = seed(real_memory, embed, "alice", "alice planted tomatoes garden")
        b = seed(real_memory, embed, "alice", "alice watered tomatoes daily")
        c = seed(real_memory, embed, "alice", "alice favourite colour blue")

        original = real_memory.episodic.grounded_retrieve_scored

        def episodic_misses_b(*args, **kwargs):
            return [
                (score, ep) for score, ep in original(*args, **kwargs)
                if ep.episode_id != b
            ]

        monkeypatch.setattr(
            real_memory.episodic, "grounded_retrieve_scored",
            episodic_misses_b,
        )
        return SimpleNamespace(a=a, b=b, c=c)

    def test_irrelevant_memory_not_retrieved_or_reinforced(
        self, real_memory, embed, seeded
    ):
        planner, llm = make_real_planner(real_memory, embed)

        result = planner.plan(self.QUERY, agent_id="alice")

        fused_ids = {fused_id(m) for m in result["planner_memories"]["fused_memory"]}
        assert seeded.c not in fused_ids
        ep = real_memory.episodic.get_by_id(seeded.c)
        assert ep.review_count == 0
        assert ep.agent_feedback == pytest.approx(0.5)
        assert real_memory.semantic.get(seeded.c).reinforcement_count == 0

    def test_fused_memory_mixes_both_types(self, real_memory, embed, seeded):
        fused = real_memory.store.retrieve(
            agent_id="alice",
            query=self.QUERY,
            embedding=embed(self.QUERY),
        )["fused_memory"]

        by_id = {fused_id(m): m for m in fused}
        assert isinstance(by_id[seeded.a], Episode)
        assert isinstance(by_id[seeded.b], SemanticMemory)
        assert not hasattr(by_id[seeded.a], "memory_id")
        assert not hasattr(by_id[seeded.b], "episode_id")

    def test_fused_memory_has_no_duplicate_ids(
        self, real_memory, embed, seeded
    ):
        """
        Guards the fusion dedup bug: Episode has no `.memory_id`, so
        dedup used to fall back to content for episodes and id for
        semantic hits — the same memory then appeared twice.
        """
        fused = real_memory.store.retrieve(
            agent_id="alice",
            query=self.QUERY,
            embedding=embed(self.QUERY),
        )["fused_memory"]

        ids = [fused_id(m) for m in fused]
        assert len(ids) == len(set(ids))

    def test_both_types_reinforced_exactly_once(
        self, real_memory, embed, seeded
    ):
        """
        Guards debug-pass bug #3 (SemanticMemory results silently not
        reinforced) AND the fusion dedup bug (Episode results reinforced
        twice). Each retrieved memory must be reinforced once per plan,
        in both tiers.
        """
        planner, _ = make_real_planner(real_memory, embed)

        planner.plan(self.QUERY, agent_id="alice")

        for mem_id in (seeded.a, seeded.b):
            ep = real_memory.episodic.get_by_id(mem_id)
            sem = real_memory.semantic.get(mem_id)
            assert ep.review_count == 1, mem_id
            assert sem.reinforcement_count == 1, mem_id

    def test_both_types_get_one_positive_feedback_step(
        self, real_memory, embed, seeded
    ):
        planner, _ = make_real_planner(real_memory, embed)

        planner.plan(self.QUERY, agent_id="alice")

        # one EMA step from 0.5 toward 1.0 with alpha=0.3
        for mem_id in (seeded.a, seeded.b):
            fb = real_memory.episodic.get_by_id(mem_id).agent_feedback
            assert fb == pytest.approx(0.65), mem_id

    def test_reinforcement_raises_stability(self, real_memory, embed, seeded):
        planner, _ = make_real_planner(real_memory, embed)
        before = real_memory.episodic.get_by_id(seeded.b).stability_hours

        planner.plan(self.QUERY, agent_id="alice")

        after = real_memory.episodic.get_by_id(seeded.b).stability_hours
        assert after > before

    def test_reinforcement_attributed_to_conversation_agent(
        self, real_memory, embed, seeded
    ):
        planner, _ = make_real_planner(real_memory, embed)

        with patch.object(
            real_memory.store,
            "reinforce",
            wraps=real_memory.store.reinforce,
        ) as spy:
            planner.plan(self.QUERY, agent_id="alice")

        assert spy.call_count >= 2
        for c in spy.call_args_list:
            assert c.kwargs["agent_id"] == "alice"

    def test_other_agents_memories_untouched(self, real_memory, embed, seeded):
        bob = seed(real_memory, embed, "bob", "bob tomatoes garden too")
        planner, _ = make_real_planner(real_memory, embed)

        planner.plan(self.QUERY, agent_id="alice")

        ep = real_memory.episodic.get_by_id(bob)
        assert ep.review_count == 0
        assert ep.agent_feedback == pytest.approx(0.5)
        assert real_memory.semantic.get(bob).reinforcement_count == 0

    def test_one_failure_does_not_stop_the_rest(
        self, real_memory, embed, seeded
    ):
        planner, _ = make_real_planner(real_memory, embed)
        real_reinforce = real_memory.store.reinforce

        def flaky(agent_id, memory_id):
            if memory_id == seeded.a:
                raise RuntimeError("boom")
            return real_reinforce(agent_id=agent_id, memory_id=memory_id)

        with patch.object(real_memory.store, "reinforce", side_effect=flaky):
            result = planner.plan(self.QUERY, agent_id="alice")

        assert result["response"]
        assert real_memory.episodic.get_by_id(seeded.b).review_count == 1

    def test_object_without_any_id_is_skipped(self):
        store = MagicMock()
        store.retrieve.return_value = {
            "fused_memory": [SimpleNamespace(content="no id here")]
        }
        executor = MagicMock()
        executor.execute.return_value = {"response": "ok"}
        planner = PlannerAgent(
            memory_store=store, executor=executor, embed_fn=lambda t: [0.0]
        )

        planner.plan("remember this", agent_id="alice")

        store.reinforce.assert_not_called()
        store.record_feedback.assert_not_called()


# ─── plan() end-to-end on real memory ──────────────────────

class TestPlanEndToEnd:

    def test_empty_task_rejected(self, real_memory, embed):
        planner, _ = make_real_planner(real_memory, embed)
        with pytest.raises(ValueError, match="task cannot be empty"):
            planner.plan("   ", agent_id="alice")

    def test_result_shape(self, real_memory, embed):
        planner, _ = make_real_planner(real_memory, embed)
        result = planner.plan("remember my cat is Miso", agent_id="alice")

        assert result["response"] == "Okay, noted."
        assert result["intent"] == TaskIntent.MEMORY
        assert result["needs_memory"] is True
        assert set(result["planner_memories"]) >= {
            "working_memory", "episodic_memory",
            "semantic_memory", "fused_memory",
        }
        assert "retrieved_memories" in result

    def test_second_turn_sees_first_turn_in_working_memory(
        self, real_memory, embed
    ):
        """
        Guards debug-pass bugs #1 and #2: working memory written to the
        right agent bucket, and the executor reading that bucket (not
        its own "executor_agent" one).
        """
        planner, llm = make_real_planner(real_memory, embed)

        planner.plan("I am allergic to penicillin", agent_id="rajan")
        planner.plan(
            "what do you remember about my penicillin allergy?",
            agent_id="rajan",
        )

        second_ctx = llm.last_context
        convo = [m for m in second_ctx if m["role"] != "system"]
        assert {"role": "user",
                "content": "I am allergic to penicillin"} in convo
        assert {"role": "assistant", "content": "Okay, noted."} in convo

    def test_second_turn_sees_first_turn_in_retrieved_memory(
        self, real_memory, embed
    ):
        planner, llm = make_real_planner(real_memory, embed)

        planner.plan("I am allergic to penicillin", agent_id="rajan")
        result = planner.plan(
            "what do you remember about my penicillin allergy?",
            agent_id="rajan",
        )

        planner_contents = [
            m.content for m in result["planner_memories"]["fused_memory"]
        ]
        assert "I am allergic to penicillin" in planner_contents

        system = [m for m in llm.last_context if m["role"] == "system"]
        assert system
        assert "I am allergic to penicillin" in system[0]["content"]

    def test_working_memory_lands_in_conversation_bucket_only(
        self, real_memory, embed
    ):
        planner, _ = make_real_planner(real_memory, embed)

        planner.plan("I am allergic to penicillin", agent_id="rajan")

        wm = real_memory.working
        assert [m["content"] for m in wm.as_messages(agent_id="rajan")] == [
            "I am allergic to penicillin",
            "Okay, noted.",
        ]
        for wrong in ("default", "planner_agent", "executor_agent"):
            assert wm.is_empty(wrong), wrong

    def test_long_term_memory_scoped_to_conversation_agent(
        self, real_memory, embed
    ):
        planner, _ = make_real_planner(real_memory, embed)

        planner.plan("I am allergic to penicillin", agent_id="rajan")

        episodes = real_memory.episodic.get_all()
        assert episodes
        assert {ep.agent_id for ep in episodes} == {"rajan"}

    def test_agents_do_not_see_each_other(self, real_memory, embed):
        planner, llm = make_real_planner(real_memory, embed)

        planner.plan("alice secret code is 4471", agent_id="alice")
        result = planner.plan(
            "what is the secret code?", agent_id="bob"
        )

        flat = " ".join(m["content"] for m in llm.last_context)
        assert "4471" not in flat
        assert result["planner_memories"]["fused_memory"] == []

    def test_question_not_stored_but_answer_is(self, real_memory, embed):
        planner, _ = make_real_planner(real_memory, embed)

        planner.plan("what is my blood type?", agent_id="rajan")

        contents = [ep.content for ep in real_memory.episodic.get_all()]
        assert "what is my blood type?" not in contents
        assert "Okay, noted." in contents
