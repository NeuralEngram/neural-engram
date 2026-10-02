# tests/test_feedback.py
"""
Phase 1 feedback loop: record_feedback / record_task_outcome.

Tested at three layers, all on REAL memory objects (conftest.real_memory):
  EpisodicMemoryStore  → the EMA update + SQLite persistence
  MemoryStore          → the public API agents and endpoints use
  POST /feedback/...   → the HTTP endpoint (conftest.api_app)

EMA reference (alpha = 0.3, start = 0.5):
  after n positive  → 1 - 0.5 * 0.7**n
  after n negative  →     0.5 * 0.7**n
"""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

import memory.episodic_memory as episodic_module


ALPHA = 0.3
UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"


def seed(real_memory, embed, agent_id="alice", content="alice grows tomatoes"):
    return real_memory.store.store(
        agent_id=agent_id,
        content=content,
        embedding=embed(content),
    )["memory_id"]


def expected_after(n, positive, start=0.5):
    return 1 - (1 - start) * (1 - ALPHA) ** n if positive \
        else start * (1 - ALPHA) ** n


def age(real_memory, memory_id, days):
    ep = real_memory.episodic.get_by_id(memory_id)
    ep.last_reviewed_at = ep.last_reviewed_at - timedelta(days=days)
    real_memory.episodic._save_episode(ep)
    return ep


# ─── Single updates ────────────────────────────────────────

class TestSingleUpdate:

    @pytest.mark.parametrize("positive, expected", [(True, 0.65), (False, 0.35)])
    def test_feedback_moves_one_ema_step(
        self, real_memory, embed, positive, expected
    ):
        mid = seed(real_memory, embed)

        assert real_memory.store.record_feedback(
            agent_id="alice", memory_id=mid, positive=positive
        ) is True

        ep = real_memory.episodic.get_by_id(mid)
        assert ep.agent_feedback == pytest.approx(expected)

    @pytest.mark.parametrize("success, expected", [(True, 0.65), (False, 0.35)])
    def test_outcome_moves_one_ema_step(
        self, real_memory, embed, success, expected
    ):
        mid = seed(real_memory, embed)

        assert real_memory.store.record_task_outcome(
            agent_id="alice", memory_id=mid, success=success
        ) is True

        ep = real_memory.episodic.get_by_id(mid)
        assert ep.task_success_rate == pytest.approx(expected)

    def test_feedback_and_outcome_are_independent(self, real_memory, embed):
        mid = seed(real_memory, embed)

        real_memory.store.record_feedback(
            agent_id="alice", memory_id=mid, positive=True
        )
        ep = real_memory.episodic.get_by_id(mid)
        assert ep.task_success_rate == pytest.approx(0.5)

        real_memory.store.record_task_outcome(
            agent_id="alice", memory_id=mid, success=False
        )
        assert ep.agent_feedback == pytest.approx(0.65)
        assert ep.task_success_rate == pytest.approx(0.35)

    def test_only_target_memory_changes(self, real_memory, embed):
        target = seed(real_memory, embed)
        other = seed(real_memory, embed, content="alice likes jazz")

        real_memory.store.record_feedback(
            agent_id="alice", memory_id=target, positive=True
        )

        ep = real_memory.episodic.get_by_id(other)
        assert ep.agent_feedback == pytest.approx(0.5)

    def test_positive_feedback_raises_priority_score(self, real_memory, embed):
        """The whole point of Phase 1: feedback must reach the DMS score."""
        a = seed(real_memory, embed, content="alice grows tomatoes")
        b = seed(real_memory, embed, content="alice grows peppers")
        ep_a = real_memory.episodic.get_by_id(a)
        ep_b = real_memory.episodic.get_by_id(b)
        ep_b.last_reviewed_at = ep_a.last_reviewed_at  # same retention
        assert ep_a.importance == ep_b.importance

        for _ in range(3):
            real_memory.store.record_feedback(
                agent_id="alice", memory_id=a, positive=True
            )
            real_memory.store.record_task_outcome(
                agent_id="alice", memory_id=a, success=True
            )

        now = ep_a.last_reviewed_at
        assert ep_a.priority_score(now) > ep_b.priority_score(now)

    def test_updates_are_persisted_to_sqlite(self, real_memory, embed):
        mid = seed(real_memory, embed)
        real_memory.store.record_feedback(
            agent_id="alice", memory_id=mid, positive=True
        )
        real_memory.store.record_task_outcome(
            agent_id="alice", memory_id=mid, success=False
        )

        # A brand-new store instance reads straight from the DB file.
        reloaded = episodic_module.EpisodicMemoryStore().get_by_id(mid)

        assert reloaded.agent_feedback == pytest.approx(0.65)
        assert reloaded.task_success_rate == pytest.approx(0.35)


# ─── Unknown / forgotten ids ───────────────────────────────

class TestMissingMemory:

    def test_unknown_id_returns_false_not_raise(self, real_memory, embed):
        seed(real_memory, embed)

        assert real_memory.store.record_feedback(
            agent_id="alice", memory_id=UNKNOWN_ID, positive=True
        ) is False
        assert real_memory.store.record_task_outcome(
            agent_id="alice", memory_id=UNKNOWN_ID, success=True
        ) is False

    def test_unknown_id_modifies_nothing(self, real_memory, embed):
        mid = seed(real_memory, embed)

        real_memory.store.record_feedback(
            agent_id="alice", memory_id=UNKNOWN_ID, positive=True
        )

        ep = real_memory.episodic.get_by_id(mid)
        assert ep.agent_feedback == pytest.approx(0.5)
        assert len(real_memory.episodic.get_all()) == 1

    def test_faded_but_unpruned_memory_accepts_feedback_but_is_not_revived(
        self, real_memory, embed
    ):
        """
        Characterisation: a memory whose retention is already below the
        forgetting threshold, but hasn't been pruned yet, still accepts
        feedback — but feedback does NOT touch retention (only recall
        does), so it stays forgotten and the next cleanup removes it.
        """
        mid = seed(real_memory, embed)
        ep = age(real_memory, mid, days=60)
        assert ep.is_forgotten()

        assert real_memory.store.record_feedback(
            agent_id="alice", memory_id=mid, positive=True
        ) is True
        assert ep.agent_feedback == pytest.approx(0.65)
        assert ep.is_forgotten()

        assert real_memory.store.cleanup() == 1
        assert real_memory.episodic.get_by_id(mid) is None

    def test_pruned_memory_returns_false(self, real_memory, embed):
        mid = seed(real_memory, embed)
        age(real_memory, mid, days=60)
        real_memory.store.cleanup()

        assert real_memory.store.record_feedback(
            agent_id="alice", memory_id=mid, positive=True
        ) is False
        assert real_memory.store.record_task_outcome(
            agent_id="alice", memory_id=mid, success=True
        ) is False
        assert real_memory.semantic.get(mid) is None

    def test_feedback_does_not_resurrect_pruned_row_in_db(
        self, real_memory, embed
    ):
        mid = seed(real_memory, embed)
        age(real_memory, mid, days=60)
        real_memory.store.cleanup()

        real_memory.store.record_feedback(
            agent_id="alice", memory_id=mid, positive=True
        )

        reloaded = episodic_module.EpisodicMemoryStore()
        assert reloaded.get_by_id(mid) is None

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known gap, needs a design decision: MemoryStore."
            "record_feedback ignores agent_id, so any agent/user can "
            "move the scores of another agent's memory. If you add an "
            "ownership check, this starts passing — remove the marker."
        ),
    )
    def test_feedback_from_non_owner_is_rejected(self, real_memory, embed):
        bobs = seed(real_memory, embed, agent_id="bob", content="bob swims")

        assert real_memory.store.record_feedback(
            agent_id="alice", memory_id=bobs, positive=False
        ) is False


# ─── Repeated calls converge ───────────────────────────────

class TestConvergence:

    def test_repeated_positive_converges_to_one(self, real_memory, embed):
        mid = seed(real_memory, embed)
        ep = real_memory.episodic.get_by_id(mid)
        history = []

        for n in range(1, 51):
            real_memory.store.record_feedback(
                agent_id="alice", memory_id=mid, positive=True
            )
            history.append(ep.agent_feedback)
            assert ep.agent_feedback == pytest.approx(expected_after(n, True))

        assert all(b >= a for a, b in zip(history, history[1:]))
        assert all(0.0 <= v <= 1.0 for v in history)
        assert history[-1] > 0.9999

    def test_repeated_negative_converges_to_zero(self, real_memory, embed):
        mid = seed(real_memory, embed)
        ep = real_memory.episodic.get_by_id(mid)

        for n in range(1, 51):
            real_memory.store.record_task_outcome(
                agent_id="alice", memory_id=mid, success=False
            )
            assert ep.task_success_rate == pytest.approx(
                expected_after(n, False)
            )
            assert ep.task_success_rate >= 0.0

        assert ep.task_success_rate < 1e-4

    def test_alternating_settles_into_bounded_cycle(self, real_memory, embed):
        """
        +,-,+,-… must settle into a fixed 2-cycle, not drift or grow.
        Fixed points for alpha=0.3: high = 0.3/0.51 ≈ 0.588,
        low = 0.7 * high ≈ 0.412.
        """
        mid = seed(real_memory, embed)
        ep = real_memory.episodic.get_by_id(mid)
        values = []

        for i in range(100):
            real_memory.store.record_feedback(
                agent_id="alice", memory_id=mid, positive=(i % 2 == 0)
            )
            values.append(ep.agent_feedback)

        high = 0.3 / (1 - 0.49)
        low = 0.7 * high
        assert values[-2] == pytest.approx(high, abs=1e-6)
        assert values[-1] == pytest.approx(low, abs=1e-6)
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_recovers_after_sign_flip(self, real_memory, embed):
        """Saturating at ~1.0 must not make the score sticky."""
        mid = seed(real_memory, embed)
        ep = real_memory.episodic.get_by_id(mid)

        for _ in range(30):
            real_memory.store.record_feedback(
                agent_id="alice", memory_id=mid, positive=True
            )
        for _ in range(30):
            real_memory.store.record_feedback(
                agent_id="alice", memory_id=mid, positive=False
            )

        assert ep.agent_feedback < 0.01

    def test_convergence_survives_restart(self, real_memory, embed):
        mid = seed(real_memory, embed)
        for _ in range(5):
            real_memory.store.record_feedback(
                agent_id="alice", memory_id=mid, positive=True
            )

        fresh = episodic_module.EpisodicMemoryStore()
        for _ in range(5):
            fresh.record_feedback(episode_id=mid, positive=True)

        assert fresh.get_by_id(mid).agent_feedback == pytest.approx(
            expected_after(10, True)
        )


# ─── POST /feedback/{agent_id}/{memory_id} ─────────────────

class TestFeedbackEndpoint:

    @pytest.fixture
    def client(self, api_app):
        return TestClient(api_app.app)

    def url(self, mid, agent="alice"):
        return f"/feedback/{agent}/{mid}"

    def test_requires_api_key(self, client, real_memory, embed):
        mid = seed(real_memory, embed)
        resp = client.post(self.url(mid), json={"positive": True})
        assert resp.status_code == 401
        ep = real_memory.episodic.get_by_id(mid)
        assert ep.agent_feedback == pytest.approx(0.5)

    def test_empty_body_is_400(self, client, auth_headers, real_memory, embed):
        mid = seed(real_memory, embed)
        resp = client.post(self.url(mid), json={}, headers=auth_headers)
        assert resp.status_code == 400

    def test_positive_only(self, client, auth_headers, real_memory, embed):
        mid = seed(real_memory, embed)

        resp = client.post(
            self.url(mid), json={"positive": True}, headers=auth_headers
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["feedback_updated"] is True
        assert body["outcome_updated"] is None
        ep = real_memory.episodic.get_by_id(mid)
        assert ep.agent_feedback == pytest.approx(0.65)
        assert ep.task_success_rate == pytest.approx(0.5)

    def test_success_only(self, client, auth_headers, real_memory, embed):
        mid = seed(real_memory, embed)

        resp = client.post(
            self.url(mid), json={"success": False}, headers=auth_headers
        )

        assert resp.status_code == 200
        assert resp.json()["outcome_updated"] is True
        ep = real_memory.episodic.get_by_id(mid)
        assert ep.task_success_rate == pytest.approx(0.35)
        assert ep.agent_feedback == pytest.approx(0.5)

    def test_both_in_one_call(self, client, auth_headers, real_memory, embed):
        mid = seed(real_memory, embed)

        resp = client.post(
            self.url(mid),
            json={"positive": True, "success": True},
            headers=auth_headers,
        )

        assert resp.status_code == 200
        ep = real_memory.episodic.get_by_id(mid)
        assert ep.agent_feedback == pytest.approx(0.65)
        assert ep.task_success_rate == pytest.approx(0.65)

    def test_unknown_id_is_404(self, client, auth_headers, real_memory, embed):
        seed(real_memory, embed)
        resp = client.post(
            self.url(UNKNOWN_ID), json={"positive": True}, headers=auth_headers
        )
        assert resp.status_code == 404

    def test_pruned_memory_is_404(self, client, auth_headers, real_memory, embed):
        mid = seed(real_memory, embed)
        age(real_memory, mid, days=60)
        real_memory.store.cleanup()

        resp = client.post(
            self.url(mid),
            json={"positive": True, "success": True},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_repeated_calls_converge(
        self, client, auth_headers, real_memory, embed
    ):
        mid = seed(real_memory, embed)

        for _ in range(20):
            assert client.post(
                self.url(mid), json={"positive": True}, headers=auth_headers
            ).status_code == 200

        ep = real_memory.episodic.get_by_id(mid)
        assert ep.agent_feedback == pytest.approx(expected_after(20, True))

    def test_visible_in_metrics_endpoint(
        self, client, auth_headers, real_memory, embed
    ):
        mid = seed(real_memory, embed)
        client.post(
            self.url(mid),
            json={"positive": True, "success": False},
            headers=auth_headers,
        )

        mems = client.get("/metrics/alice", headers=auth_headers).json()
        row = next(m for m in mems["memories"] if m["id"] == mid)
        assert row["agent_feedback"] == pytest.approx(0.65)
        assert row["task_success_rate"] == pytest.approx(0.35)
