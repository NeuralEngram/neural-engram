# tests/test_integration_isolation.py
"""
Multi-agent isolation and the shared-memory path, end-to-end through a
real Orchestrator.

Both patients talk about the same things (daughter, medicine, doctor,
tablet), so similarity alone WOULD pull the other patient's memories in.
The only thing that keeps them apart is agent_id scoping in every tier.
Each patient has private marker words; after every single turn we check
that the other patient's markers appear nowhere — not in the request
sent to the LLM, not in working memory, not in any retrieval tier, not
in reinforcement side effects.
"""

import pytest

from memory.episodic_memory import Episode
from memory.semantic_memory import SemanticMemory


RAJAN_TURNS = [
    "my daughter Priya brings my medicine",
    "my doctor appointment is on monday",
    "I take my insulin tablet before breakfast",
    "my daughter Priya lives in Pune",
]
ALICE_TURNS = [
    "my daughter Meera brings my medicine",
    "my doctor appointment is on friday",
    "I take my vitamin tablet before breakfast",
    "my daughter Meera lives in Leeds",
]
MARKERS = {
    "rajan": {"Priya", "monday", "insulin", "Pune"},
    "alice": {"Meera", "friday", "vitamin", "Leeds"},
}
OTHER = {"rajan": "alice", "alice": "rajan"}
PROBE = "daughter medicine doctor appointment tablet breakfast lives"


def echo_reply(request):
    """Assistant echoes the user, so replies carry the markers too."""
    return "Noted: " + request["messages"][-1]["content"]


def leaked(text, agent):
    """Markers of the OTHER agent found in `text`."""
    return {m for m in MARKERS[OTHER[agent]] if m in text}


def owner(mem):
    if isinstance(mem, Episode):
        return mem.agent_id
    if isinstance(mem, SemanticMemory):
        return mem.metadata["agent_id"]
    raise TypeError(type(mem))


def snapshot_reinforcement(orch, agent):
    episodic = orch.memory_store.orchestrator.episodic_memory
    return {
        ep.episode_id: (ep.review_count, ep.agent_feedback)
        for ep in episodic.get_all() if ep.agent_id == agent
    }


def assert_isolated(orch, agent, embed):
    """Every tier, as seen by `agent`, contains only `agent`'s data."""
    store = orch.memory_store

    wm = store.get_context(agent, limit=50)
    for msg in wm:
        assert not leaked(msg["content"], agent), ("working", msg)

    result = store.retrieve(agent_id=agent, query=PROBE,
                            embedding=embed(PROBE), top_k=20)
    for tier in ("episodic_memory", "semantic_memory", "fused_memory"):
        for mem in result[tier]:
            assert owner(mem) == agent, (tier, mem.content)
            assert not leaked(mem.content, agent), (tier, mem.content)
    for msg in result["working_memory"]:
        assert not leaked(msg["content"], agent)

    for row in store.orchestrator.metrics(agent_id=agent)["memories"]:
        assert row["agent_id"] == agent


INTERLEAVINGS = [
    "RARARARA",   # strict alternation
    "ARARARAR",   # alternation, other agent first
    "RRRRAAAA",   # one full conversation, then the other
    "AAAARRRR",
    "RAARRAAR",   # irregular
]


@pytest.mark.parametrize("schedule", INTERLEAVINGS)
class TestInterleavedConversations:

    def run(self, orch, schedule, embed, check):
        queues = {"rajan": list(RAJAN_TURNS), "alice": list(ALICE_TURNS)}
        for step, code in enumerate(schedule):
            agent = "rajan" if code == "R" else "alice"
            task = queues[agent].pop(0)
            other = OTHER[agent]
            before_other = snapshot_reinforcement(orch, other)

            result = orch.route(task, agent_id=agent)

            check(orch, agent, result, before_other, step, embed)

    def test_llm_request_never_contains_other_agent(
        self, real_orchestrator, embed, schedule
    ):
        orch = real_orchestrator()
        orch.client.reply = echo_reply

        def check(orch, agent, result, before_other, step, embed):
            text = orch.client.last_request_text()
            assert not leaked(text, agent), (step, agent)

        self.run(orch, schedule, embed, check)

    def test_llm_request_contains_own_prior_turns(
        self, real_orchestrator, embed, schedule
    ):
        """
        Isolation must not be achieved by showing nobody anything: each
        turn's request carries that agent's own previous turns as
        conversation history (catches the executor reading the wrong
        working-memory bucket, original bug #2).
        """
        orch = real_orchestrator()
        orch.client.reply = echo_reply
        said = {"rajan": [], "alice": []}

        def check(orch, agent, result, before_other, step, embed):
            history = [
                m["content"] for m in orch.client.last_messages[:-1]
                if m["role"] == "user"
            ]
            assert history == said[agent][-5:], (step, agent)
            said[agent].append(orch.client.last_messages[-1]["content"])

        self.run(orch, schedule, embed, check)

    def test_every_tier_isolated_after_every_turn(
        self, real_orchestrator, embed, schedule
    ):
        orch = real_orchestrator()
        orch.client.reply = echo_reply

        def check(orch, agent, result, before_other, step, embed):
            assert_isolated(orch, "rajan", embed)
            assert_isolated(orch, "alice", embed)

        self.run(orch, schedule, embed, check)

    def test_route_results_only_contain_own_memories(
        self, real_orchestrator, embed, schedule
    ):
        orch = real_orchestrator()
        orch.client.reply = echo_reply

        def check(orch, agent, result, before_other, step, embed):
            for key in ("planner_memories", "retrieved_memories"):
                for mem in result[key]["fused_memory"]:
                    assert owner(mem) == agent, (key, mem.content)

        self.run(orch, schedule, embed, check)

    def test_turn_never_reinforces_other_agent(
        self, real_orchestrator, embed, schedule
    ):
        orch = real_orchestrator()

        def check(orch, agent, result, before_other, step, embed):
            after = snapshot_reinforcement(orch, OTHER[agent])
            assert after == before_other, (step, agent)

        self.run(orch, schedule, embed, check)

    def test_each_agent_sees_its_full_own_history(
        self, real_orchestrator, embed, schedule
    ):
        orch = real_orchestrator()
        orch.client.reply = echo_reply
        self.run(orch, schedule, embed, lambda *a: None)

        for agent, turns in (("rajan", RAJAN_TURNS), ("alice", ALICE_TURNS)):
            users = [m["content"] for m in
                     orch.memory_store.get_context(agent, limit=50)
                     if m["role"] == "user"]
            assert users == turns


class TestIsolationEdgeCases:

    def test_identical_content_stays_per_agent(self, real_orchestrator, embed):
        orch = real_orchestrator()
        same = "my daughter brings my medicine"

        orch.route(same, agent_id="rajan")
        orch.route(same, agent_id="alice")

        for agent in ("rajan", "alice"):
            r = orch.memory_store.retrieve(
                agent_id=agent, query=same, embedding=embed(same)
            )
            hits = [m for m in r["fused_memory"] if m.content == same]
            assert len(hits) == 1
            assert owner(hits[0]) == agent

    def test_one_agent_filling_wm_does_not_evict_other(self, real_orchestrator):
        orch = real_orchestrator()

        orch.route(RAJAN_TURNS[0], agent_id="rajan")
        for n in range(30):  # 60 messages » capacity 20
            orch.route(f"alice note {n} about gardening", agent_id="alice")

        rajan = orch.memory_store.get_context("rajan")
        assert [m["content"] for m in rajan] == [RAJAN_TURNS[0],
                                                 "Okay, noted."]

    def test_next_rajan_turn_after_long_alice_session(self, real_orchestrator):
        orch = real_orchestrator()
        orch.client.reply = echo_reply

        orch.route(RAJAN_TURNS[0], agent_id="rajan")
        for turn in ALICE_TURNS * 3:
            orch.route(turn, agent_id="alice")
        orch.route("who brings my medicine?", agent_id="rajan")

        text = orch.client.last_request_text()
        assert "Priya" in text
        assert not leaked(text, "rajan")

    def test_agent_ids_are_exact_match(self, real_orchestrator):
        """Characterisation: "Rajan" and "rajan" are different agents."""
        orch = real_orchestrator()

        orch.route(RAJAN_TURNS[0], agent_id="rajan")
        orch.route("who brings my medicine?", agent_id="Rajan")

        assert "Priya" not in orch.client.last_request_text()


# ═══════════════════════════════════════════════════════════
# Shared memory
# ═══════════════════════════════════════════════════════════
#
#   owner   = "caregiver"  stores shared=True
#   viewer  = "rajan"      should see it
#   outsider= "alice"      stores a NON-shared memory on the same topic;
#                          must never reach rajan (or caregiver)

SHARED = "rajan daughter Priya visits every sunday with medicine"
PRIVATE = "alice daughter Meera visits every sunday with medicine"
QUERY = "who visits on sunday with medicine?"


@pytest.fixture
def shared_setup(real_orchestrator, embed):
    orch = real_orchestrator()
    store = orch.memory_store
    shared_id = store.store(
        agent_id="caregiver", content=SHARED, embedding=embed(SHARED),
        context={"shared": True},
    )["memory_id"]
    private_id = store.store(
        agent_id="alice", content=PRIVATE, embedding=embed(PRIVATE),
    )["memory_id"]
    return orch, shared_id, private_id


class TestSharedMemory:

    def retrieve(self, orch, agent, embed):
        return orch.memory_store.retrieve(
            agent_id=agent, query=QUERY, embedding=embed(QUERY)
        )

    def test_shared_flag_persisted_on_episode(self, shared_setup):
        orch, shared_id, _ = shared_setup
        ep = orch.memory_store.orchestrator.episodic_memory.get_by_id(
            shared_id)
        assert ep.shared is True
        assert ep.agent_id == "caregiver"

    def test_other_agent_retrieve_sees_shared(self, shared_setup, embed):
        orch, shared_id, _ = shared_setup

        r = self.retrieve(orch, "rajan", embed)

        assert shared_id in [e.episode_id for e in r["episodic_memory"]]
        assert SHARED in [m.content for m in r["fused_memory"]]

    def test_non_shared_third_agent_does_not_leak(self, shared_setup, embed):
        orch, _, private_id = shared_setup

        for viewer in ("rajan", "caregiver"):
            r = self.retrieve(orch, viewer, embed)
            for tier in ("episodic_memory", "semantic_memory",
                         "fused_memory"):
                assert PRIVATE not in [m.content for m in r[tier]], \
                    (viewer, tier)

    def test_shared_reaches_llm_private_does_not(self, shared_setup):
        orch, _, _ = shared_setup

        orch.route(QUERY, agent_id="rajan")

        assert f"Memory: {SHARED}" in orch.client.last_system
        text = orch.client.last_request_text()
        assert "Meera" not in text

    def test_shared_does_not_enter_viewers_working_memory(self, shared_setup):
        orch, _, _ = shared_setup

        assert orch.memory_store.get_context("rajan") == []
        assert [m["content"] for m in
                orch.memory_store.get_context("caregiver")] == [SHARED]

    def test_shared_visible_to_every_agent(self, shared_setup, embed):
        orch, shared_id, _ = shared_setup
        for viewer in ("rajan", "alice", "someone_new"):
            r = self.retrieve(orch, viewer, embed)
            assert shared_id in [e.episode_id for e in r["episodic_memory"]]

    def test_owner_still_sees_own_shared(self, shared_setup, embed):
        orch, shared_id, _ = shared_setup
        r = self.retrieve(orch, "caregiver", embed)
        assert shared_id in [
            getattr(m, "episode_id", None) or m.memory_id
            for m in r["fused_memory"]
        ]

    def test_shared_in_viewers_metrics(self, shared_setup):
        orch, shared_id, private_id = shared_setup
        rows = orch.memory_store.orchestrator.metrics(agent_id="rajan")
        ids = {m["id"] for m in rows["memories"]}
        assert shared_id in ids
        assert private_id not in ids

    def test_viewer_turn_reinforces_shared_memory(self, shared_setup):
        orch, shared_id, _ = shared_setup

        orch.route(QUERY, agent_id="rajan")

        ep = orch.memory_store.orchestrator.episodic_memory.get_by_id(
            shared_id)
        assert ep.review_count == 1

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Needs a decision: SemanticMemoryStore.search_multi filters "
            "on agent_id only, so shared memories are visible to other "
            "agents through the EPISODIC tier only (and only if they pass "
            "its similarity threshold). The semantic tier never returns "
            "them. Remove this marker if you make search_multi include "
            "shared=True points."
        ),
    )
    def test_shared_visible_in_semantic_tier(self, shared_setup, embed):
        orch, shared_id, _ = shared_setup
        r = self.retrieve(orch, "rajan", embed)
        assert shared_id in [s.memory_id for s in r["semantic_memory"]]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Needs a decision: Episode.cross_agent_recall_count is never "
            "incremented anywhere, so consensus_score() is always 0 and "
            "the 8% W_CONSENSUS term in the shared-memory DMS formula "
            "is inert."
        ),
    )
    def test_cross_agent_recall_is_counted(self, shared_setup):
        orch, shared_id, _ = shared_setup
        orch.route(QUERY, agent_id="rajan")
        ep = orch.memory_store.orchestrator.episodic_memory.get_by_id(
            shared_id)
        assert ep.cross_agent_recall_count == 1
