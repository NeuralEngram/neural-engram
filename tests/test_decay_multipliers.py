# tests/test_decay_multipliers.py
"""
AGENT_DECAY_MULTIPLIERS must change how fast a given agent's memories
actually decay — at the EpisodicMemoryStore level (the multiplier stored
on each Episode and used in retention / forgetting / pruning / alerts),
not merely parse.

Effective stability = stability_hours / multiplier, so at the same age:
    R_fast = R_normal ** multiplier
"""

from datetime import datetime, timedelta, timezone

import pytest

import memory.episodic_memory as episodic_module
from memory.notifier import detect_alerts
from tests.conftest import _isolate_storage


CONTENT = "my daughter brings my medicine"  # same importance for everyone


@pytest.fixture
def make_store(tmp_path, monkeypatch):
    """make_store("rajan:3.0,...") → fresh EpisodicMemoryStore on tmp DB."""
    def factory(config=None):
        _isolate_storage(tmp_path, monkeypatch, config or "")
        if config is None:
            monkeypatch.delenv("AGENT_DECAY_MULTIPLIERS")
        return episodic_module.EpisodicMemoryStore()
    return factory


def add(store, agent_id, content=CONTENT):
    mid = f"{agent_id}-{len(store.get_all())}"
    store.add(memory_id=mid, agent_id=agent_id, content=content)
    return store.get_by_id(mid)


def age_all(store, hours):
    for ep in store.get_all():
        ep.last_reviewed_at -= timedelta(hours=hours)
        store._save_episode(ep)


# ─── Applied to stored episodes ────────────────────────────

class TestAppliedPerAgent:

    def test_multiplier_stamped_on_each_agents_episodes(self, make_store):
        store = make_store("rajan:3.0,alice:0.5")

        assert add(store, "rajan").decay_multiplier == 3.0
        assert add(store, "alice").decay_multiplier == 0.5
        assert add(store, "bob").decay_multiplier == 1.0

    def test_same_content_same_base_stability(self, make_store):
        store = make_store("rajan:3.0")
        r, b = add(store, "rajan"), add(store, "bob")
        assert r.stability_hours == b.stability_hours
        assert r.importance == b.importance

    def test_retention_follows_power_law(self, make_store):
        store = make_store("rajan:3.0,alice:0.5")
        eps = {a: add(store, a) for a in ("rajan", "alice", "bob")}
        for ep in eps.values():
            ep.last_reviewed_at = eps["bob"].last_reviewed_at
        now = eps["bob"].last_reviewed_at + timedelta(hours=30)

        r_bob = eps["bob"].retention(now)
        assert 0.0 < r_bob < 1.0
        assert eps["rajan"].retention(now) == pytest.approx(r_bob ** 3.0)
        assert eps["alice"].retention(now) == pytest.approx(r_bob ** 0.5)

    def test_only_fast_agent_forgotten_after_same_age(self, make_store):
        store = make_store("rajan:3.0")
        rajan, bob = add(store, "rajan"), add(store, "bob")

        # just past rajan's forgetting point, well before bob's
        hours = 1
        while not rajan.retention(
            datetime.now(timezone.utc) + timedelta(hours=hours)
        ) < 0.30:
            hours += 1
        age_all(store, hours)

        assert rajan.is_forgotten()
        assert not bob.is_forgotten()

    def test_prune_removes_only_fast_agent(self, make_store):
        store = make_store("rajan:3.0")
        rajan, bob = add(store, "rajan"), add(store, "bob")
        age_all(store, 50)
        assert rajan.is_forgotten() and not bob.is_forgotten()

        removed = store.prune_forgotten()

        assert removed == [rajan.episode_id]
        assert store.get_by_id(bob.episode_id) is not None

    def test_apply_decay_removes_only_fast_agent(self, make_store):
        store = make_store("rajan:3.0")
        rajan, bob = add(store, "rajan"), add(store, "bob")
        age_all(store, 50)

        store.apply_decay()

        assert store.get_by_id(rajan.episode_id) is None
        assert store.get_by_id(bob.episode_id) is not None

    def test_retrieval_drops_fast_agents_faded_memory(self, make_store, embed):
        store = make_store("rajan:3.0")
        add(store, "rajan")
        add(store, "bob")
        age_all(store, 50)

        q = embed(CONTENT)
        assert store.grounded_retrieve(CONTENT, q, agent_id="rajan") == []
        assert len(store.grounded_retrieve(CONTENT, q, agent_id="bob")) == 1

    def test_alert_fires_sooner_for_fast_agent(self, make_store):
        store = make_store("rajan:3.0")
        add(store, "rajan")
        add(store, "bob")
        age_all(store, 30)
        eps = store.get_all()

        assert detect_alerts(eps, "rajan")
        assert detect_alerts(eps, "bob") == []

    def test_multiplier_survives_reinforcement(self, make_store):
        store = make_store("rajan:3.0")
        ep = add(store, "rajan")
        before = ep.stability_hours

        store.recall(ep.episode_id)

        assert ep.decay_multiplier == 3.0
        assert ep.stability_hours > before


# ─── Persistence & when config is read ─────────────────────

class TestConfigLifecycle:

    def test_multiplier_persisted_and_reloaded(self, make_store):
        store = make_store("rajan:3.0")
        mid = add(store, "rajan").episode_id

        reloaded = make_store("rajan:3.0").get_by_id(mid)
        assert reloaded.decay_multiplier == 3.0

    def test_multiplier_fixed_at_creation_time(self, make_store):
        """
        Characterisation: the multiplier is stamped when the episode is
        created. Changing the env later affects NEW memories only.
        """
        store = make_store("rajan:3.0")
        old = add(store, "rajan").episode_id

        store2 = make_store("rajan:1.5")
        new = add(store2, "rajan")

        assert store2.get_by_id(old).decay_multiplier == 3.0
        assert new.decay_multiplier == 1.5

    def test_default_when_unset_is_rajan_demo(self, make_store):
        store = make_store(None)
        assert add(store, "rajan").decay_multiplier == 3.0
        assert add(store, "alice").decay_multiplier == 1.0

    def test_empty_string_disables_all(self, make_store):
        store = make_store("")
        assert add(store, "rajan").decay_multiplier == 1.0

    def test_whitespace_tolerated(self, make_store):
        store = make_store("  rajan : 2.5 ,  alice:4 ")
        assert add(store, "rajan").decay_multiplier == 2.5
        assert add(store, "alice").decay_multiplier == 4.0

    @pytest.mark.parametrize(
        "config",
        ["rajan:abc,alice:2", "rajan,alice:2", "rajan:0,alice:2",
         "rajan:-1,alice:2", "rajan:inf,alice:2", "rajan:nan,alice:2"],
    )
    def test_invalid_entries_skipped_valid_kept(self, make_store, config):
        store = make_store(config)
        assert add(store, "rajan").decay_multiplier == 1.0
        assert add(store, "alice").decay_multiplier == 2.0

    def test_invalid_entry_cannot_break_retrieval(self, make_store, embed):
        """One bad entry used to raise for EVERY agent's retrieval."""
        store = make_store("rajan:0")
        add(store, "rajan")
        add(store, "bob")

        q = embed(CONTENT)
        assert len(store.grounded_retrieve(CONTENT, q, agent_id="bob")) == 1
        assert detect_alerts(store.get_all(), "rajan") == []


# ─── Through the real Orchestrator ─────────────────────────

class TestThroughOrchestrator:

    def test_route_stamps_conversation_agents_multiplier(
        self, real_orchestrator
    ):
        orch = real_orchestrator(decay_multipliers="rajan:3.0")

        orch.route(CONTENT, agent_id="rajan")
        orch.route(CONTENT, agent_id="alice")

        mults = {
            (ep.agent_id, ep.decay_multiplier)
            for ep in orch.memory_store.orchestrator.episodic_memory.get_all()
        }
        assert mults == {("rajan", 3.0), ("alice", 1.0)}

    def test_scheduler_cycle_forgets_only_fast_agent(self, real_orchestrator):
        orch = real_orchestrator(decay_multipliers="rajan:3.0")
        orch.route(CONTENT, agent_id="rajan")
        orch.route(CONTENT, agent_id="alice")
        episodic = orch.memory_store.orchestrator.episodic_memory
        age_all(episodic, 50)

        orch.scheduler.run_once()

        assert {ep.agent_id for ep in episodic.get_all()} == {"alice"}
