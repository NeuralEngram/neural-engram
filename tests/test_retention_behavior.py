# tests/test_retention_behaviour.py

"""
End-to-end retention behaviour tests.

Verifies that memories correctly:
- decay over time
- reinforce after recall
- get marked forgotten below threshold
- show diminishing returns on repeat reinforcement
- survive long gaps when stability is high
- expire faster when stability is low
"""

import math
import pytest
from datetime import datetime, timezone, timedelta
from memory.ebbinghaus import (
    EbbinghausEngine,
    RetentionState,
    DEFAULT_RETENTION_THRESHOLD,
    BASE_STABILITY_HOURS,
    MAX_STABILITY_HOURS,
)
from memory.episodic_memory import Episode


# ─── Helpers ───────────────────────────────────────────────

def fixed_clock(t):
    return lambda: t

def hours_to_seconds(h):
    return h * 3600

def make_engine(clock):
    return EbbinghausEngine(clock=clock)

def make_state(clock, stability_hours=BASE_STABILITY_HOURS):
    engine = make_engine(clock)
    return engine.create_state(stability_hours=stability_hours)

NOW = 1_000_000.0  # fixed epoch


# ═══════════════════════════════════════════════════════════
# 1. DECAY BEHAVIOUR
# ═══════════════════════════════════════════════════════════

class TestDecayBehaviour:

    def test_fresh_memory_has_full_retention(self):
        """A memory accessed right now should have retention = 1.0"""
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state()
        decayed = engine.decay(state)
        assert decayed.retention == pytest.approx(1.0)

    def test_retention_drops_below_one_after_time_passes(self):
        """Any elapsed time should reduce retention."""
        state = make_state(fixed_clock(NOW))
        engine = make_engine(fixed_clock(NOW + hours_to_seconds(1)))
        decayed = engine.decay(state)
        assert decayed.retention < 1.0

    def test_retention_at_one_stability_period(self):
        """At t = S, retention should equal e^-1 ≈ 0.368"""
        state = make_state(fixed_clock(NOW), stability_hours=BASE_STABILITY_HOURS)
        engine = make_engine(
            fixed_clock(NOW + hours_to_seconds(BASE_STABILITY_HOURS))
        )
        decayed = engine.decay(state)
        assert decayed.retention == pytest.approx(math.exp(-1), rel=1e-4)

    def test_retention_at_two_stability_periods(self):
        """At t = 2S, retention should equal e^-2 ≈ 0.135"""
        state = make_state(fixed_clock(NOW), stability_hours=BASE_STABILITY_HOURS)
        engine = make_engine(
            fixed_clock(NOW + hours_to_seconds(BASE_STABILITY_HOURS * 2))
        )
        decayed = engine.decay(state)
        assert decayed.retention == pytest.approx(math.exp(-2), rel=1e-4)

    def test_retention_monotonically_decreases(self):
        """Retention should strictly decrease as time increases."""
        state = make_state(fixed_clock(NOW))
        retentions = []
        for hours in [0, 6, 12, 24, 48, 96, 200]:
            engine = make_engine(fixed_clock(NOW + hours_to_seconds(hours)))
            retentions.append(engine.decay(state).retention)
        assert all(
            retentions[i] >= retentions[i + 1]
            for i in range(len(retentions) - 1)
        )

    def test_high_stability_decays_slower(self):
        """Higher stability_hours means slower decay at same elapsed time."""
        elapsed = hours_to_seconds(48)
        state_low = make_state(fixed_clock(NOW), stability_hours=24.0)
        state_high = make_state(fixed_clock(NOW), stability_hours=200.0)
        engine = make_engine(fixed_clock(NOW + elapsed))
        r_low = engine.decay(state_low).retention
        r_high = engine.decay(state_high).retention
        assert r_high > r_low

    def test_low_stability_decays_faster(self):
        """Low stability memory should be nearly forgotten quickly."""
        state = make_state(fixed_clock(NOW), stability_hours=1.0)
        engine = make_engine(fixed_clock(NOW + hours_to_seconds(10)))
        decayed = engine.decay(state)
        assert decayed.retention < 0.1

    def test_very_old_memory_near_zero(self):
        """Memory from a year ago should be essentially forgotten."""
        state = make_state(fixed_clock(NOW))
        engine = make_engine(fixed_clock(NOW + hours_to_seconds(8760)))
        decayed = engine.decay(state)
        assert decayed.retention < 0.01

    def test_decay_does_not_change_stability(self):
        """Decay should never alter stability_hours."""
        state = make_state(fixed_clock(NOW))
        engine = make_engine(fixed_clock(NOW + hours_to_seconds(100)))
        decayed = engine.decay(state)
        assert decayed.stability_hours == state.stability_hours

    def test_decay_does_not_change_access_count(self):
        """Decay is passive — it should not increment access_count."""
        state = make_state(fixed_clock(NOW))
        engine = make_engine(fixed_clock(NOW + hours_to_seconds(10)))
        decayed = engine.decay(state)
        assert decayed.access_count == 0


# ═══════════════════════════════════════════════════════════
# 2. FORGOTTEN THRESHOLD
# ═══════════════════════════════════════════════════════════

class TestForgottenThreshold:

    def test_not_forgotten_when_fresh(self):
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state()
        assert engine.forgotten(state) is False

    def test_forgotten_after_extreme_time(self):
        state = make_state(fixed_clock(NOW))
        engine = make_engine(fixed_clock(NOW + hours_to_seconds(100_000)))
        assert engine.forgotten(state) is True

    def test_forgotten_exactly_at_threshold(self):
        """At exactly the threshold time, memory should be forgotten."""
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state(stability_hours=BASE_STABILITY_HOURS)
        threshold_hours = -BASE_STABILITY_HOURS * math.log(
            DEFAULT_RETENTION_THRESHOLD
        )
        # just past threshold
        engine_future = make_engine(
            fixed_clock(NOW + hours_to_seconds(threshold_hours + 0.01))
        )
        assert engine_future.forgotten(state) is True

    def test_not_forgotten_just_before_threshold(self):
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state(stability_hours=BASE_STABILITY_HOURS)
        threshold_hours = -BASE_STABILITY_HOURS * math.log(
            DEFAULT_RETENTION_THRESHOLD
        )
        # just before threshold
        engine_future = make_engine(
            fixed_clock(NOW + hours_to_seconds(threshold_hours - 0.01))
        )
        assert engine_future.forgotten(state) is False

    def test_custom_threshold_respected(self):
        """Higher threshold means forgotten sooner."""
        state = make_state(fixed_clock(NOW), stability_hours=BASE_STABILITY_HOURS)
        elapsed = hours_to_seconds(BASE_STABILITY_HOURS * 0.5)
        engine_strict = EbbinghausEngine(
            retention_threshold=0.8,
            clock=fixed_clock(NOW + elapsed)
        )
        engine_lenient = EbbinghausEngine(
            retention_threshold=0.1,
            clock=fixed_clock(NOW + elapsed)
        )
        assert engine_strict.forgotten(state) is True
        assert engine_lenient.forgotten(state) is False


# ═══════════════════════════════════════════════════════════
# 3. REINFORCEMENT BEHAVIOUR
# ═══════════════════════════════════════════════════════════

class TestReinforcementBehaviour:

    def test_reinforcement_resets_retention_to_one(self):
        """After reinforcement, retention must be 1.0 regardless of decay."""
        state = make_state(fixed_clock(NOW))
        engine = make_engine(fixed_clock(NOW + hours_to_seconds(20)))
        decayed = engine.decay(state)
        assert decayed.retention < 1.0
        reinforced = engine.reinforce(decayed)
        assert reinforced.retention == pytest.approx(1.0)

    def test_reinforcement_increases_stability(self):
        """Each reinforcement should grow stability_hours."""
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state()
        reinforced = engine.reinforce(state, quality=1.0)
        assert reinforced.stability_hours > state.stability_hours

    def test_reinforcement_increments_access_count(self):
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state()
        reinforced = engine.reinforce(state)
        assert reinforced.access_count == 1

    def test_multiple_reinforcements_grow_stability(self):
        """Stability should keep growing with each reinforcement."""
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state()
        stabilities = [state.stability_hours]
        for _ in range(5):
            state = engine.reinforce(state, quality=1.0)
            stabilities.append(state.stability_hours)
        assert all(
            stabilities[i] < stabilities[i + 1]
            for i in range(len(stabilities) - 1)
        )

    def test_low_quality_reinforcement_less_boost(self):
        """quality=0.1 should boost less than quality=1.0."""
        engine = make_engine(fixed_clock(NOW))
        s1 = engine.create_state()
        s2 = engine.create_state()
        r_low = engine.reinforce(s1, quality=0.1)
        r_high = engine.reinforce(s2, quality=1.0)
        assert r_high.stability_hours > r_low.stability_hours

    def test_quality_zero_no_stability_change(self):
        """quality=0.0 means no boost — stability stays the same."""
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state()
        reinforced = engine.reinforce(state, quality=0.0)
        assert reinforced.stability_hours == pytest.approx(state.stability_hours)

    def test_diminishing_returns_on_multiplier(self):
        """Each successive boost multiplier should shrink."""
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state()
        multipliers = []
        for _ in range(6):
            prev = state.stability_hours
            state = engine.reinforce(state, quality=1.0)
            multipliers.append(state.stability_hours / prev)
        assert all(
            multipliers[i] > multipliers[i + 1]
            for i in range(len(multipliers) - 1)
        )

    def test_stability_never_exceeds_max(self):
        """Reinforcing many times should never break the MAX_STABILITY_HOURS cap."""
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state(stability_hours=MAX_STABILITY_HOURS)
        for _ in range(10):
            state = engine.reinforce(state, quality=1.0)
        assert state.stability_hours <= MAX_STABILITY_HOURS


# ═══════════════════════════════════════════════════════════
# 4. TIME UNTIL FORGOTTEN
# ═══════════════════════════════════════════════════════════

class TestTimeUntilForgotten:

    def test_fresh_memory_has_long_horizon(self):
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state()
        remaining = engine.time_until_forgotten(state)
        assert remaining > 0

    def test_already_forgotten_returns_zero(self):
        state = make_state(fixed_clock(NOW))
        engine = make_engine(fixed_clock(NOW + hours_to_seconds(1_000_000)))
        assert engine.time_until_forgotten(state) == 0.0

    def test_higher_stability_longer_horizon(self):
        engine_now = make_engine(fixed_clock(NOW))
        s_low = engine_now.create_state(stability_hours=24.0)
        s_high = engine_now.create_state(stability_hours=500.0)
        assert engine_now.time_until_forgotten(s_high) > \
               engine_now.time_until_forgotten(s_low)

    def test_horizon_shrinks_as_time_passes(self):
        """The remaining time should decrease as the clock advances."""
        state = make_state(fixed_clock(NOW))
        remaining = []
        for hours in [0, 5, 10, 20]:
            engine = make_engine(fixed_clock(NOW + hours_to_seconds(hours)))
            remaining.append(engine.time_until_forgotten(state))
        assert all(
            remaining[i] >= remaining[i + 1]
            for i in range(len(remaining) - 1)
        )

    def test_reinforcement_extends_horizon(self):
        """After reinforcement, the time until forgotten should increase."""
        engine = make_engine(fixed_clock(NOW + hours_to_seconds(10)))
        state = make_state(fixed_clock(NOW))
        before = engine.time_until_forgotten(state)
        reinforced = engine.reinforce(state)
        after = engine.time_until_forgotten(reinforced)
        assert after > before


# ═══════════════════════════════════════════════════════════
# 5. EPISODE RETENTION INTEGRATION
# ═══════════════════════════════════════════════════════════

class TestEpisodeRetentionBehaviour:

    def _make_episode(self, last_reviewed_at, stability_hours=BASE_STABILITY_HOURS):
        return Episode(
            content="test memory",
            context={"role": "user"},
            agent_id="agent_1",
            stability_hours=stability_hours,
            last_reviewed_at=last_reviewed_at,
        )

    def test_fresh_episode_not_forgotten(self):
        now = datetime.now(timezone.utc)
        ep = self._make_episode(last_reviewed_at=now)
        assert ep.is_forgotten(now=now) is False

    def test_old_episode_is_forgotten(self):
        old = datetime.now(timezone.utc) - timedelta(hours=100_000)
        ep = self._make_episode(last_reviewed_at=old)
        assert ep.is_forgotten() is True

    def test_episode_retention_decays_over_time(self):
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=BASE_STABILITY_HOURS)
        ep = self._make_episode(last_reviewed_at=past)
        r = ep.retention(now=now)
        assert r == pytest.approx(math.exp(-1), rel=1e-3)

    def test_high_stability_episode_survives_long_gap(self):
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=200)
        ep = self._make_episode(
            last_reviewed_at=past,
            stability_hours=MAX_STABILITY_HOURS
        )
        assert ep.is_forgotten(now=now) is False

    def test_low_stability_episode_forgotten_quickly(self):
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=5)
        ep = self._make_episode(
            last_reviewed_at=past,
            stability_hours=1.0
        )
        assert ep.is_forgotten(now=now) is True

    def test_retention_always_between_zero_and_one(self):
        now = datetime.now(timezone.utc)
        for hours_ago in [0, 1, 10, 100, 10000]:
            past = now - timedelta(hours=hours_ago)
            ep = self._make_episode(last_reviewed_at=past)
            r = ep.retention(now=now)
            assert 0.0 <= r <= 1.0


# ═══════════════════════════════════════════════════════════
# 6. FULL LIFECYCLE SCENARIO
# ═══════════════════════════════════════════════════════════

class TestFullLifecycleScenario:

    def test_memory_survives_reinforcement_then_long_gap(self):
        """
        Simulate: learn → reinforce 3x → long gap → check still alive.
        """
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state(stability_hours=BASE_STABILITY_HOURS)

        # reinforce 3 times
        for _ in range(3):
            state = engine.reinforce(state, quality=1.0)

        # jump forward — but less than the new extended horizon
        new_horizon = engine.time_until_forgotten(state)
        engine_future = make_engine(
            fixed_clock(NOW + hours_to_seconds(new_horizon * 0.5))
        )
        assert engine_future.forgotten(state) is False

    def test_unreinforced_memory_forgotten_eventually(self):
        """
        Simulate: learn → never review → eventually forgotten.
        """
        state = make_state(fixed_clock(NOW), stability_hours=BASE_STABILITY_HOURS)
        threshold_hours = -BASE_STABILITY_HOURS * math.log(
            DEFAULT_RETENTION_THRESHOLD
        )
        engine_future = make_engine(
            fixed_clock(NOW + hours_to_seconds(threshold_hours * 2))
        )
        assert engine_future.forgotten(state) is True

    def test_forgotten_memory_revived_by_reinforcement(self):
        """
        Simulate: forget → encounter again → retention restored.
        """
        state = make_state(fixed_clock(NOW))
        engine_old = make_engine(fixed_clock(NOW + hours_to_seconds(100_000)))
        assert engine_old.forgotten(state) is True

        # re-encounter resets retention
        revived = engine_old.reinforce(state, quality=1.0)
        assert revived.retention == pytest.approx(1.0)

    def test_importance_and_priority_extend_survival(self):
        """
        High importance + high priority → much higher stability → survives longer.
        """
        engine = make_engine(fixed_clock(NOW))
        state_normal = engine.create_state(
            stability_hours=BASE_STABILITY_HOURS,
            importance=1.0,
            priority=1
        )
        state_important = engine.create_state(
            stability_hours=BASE_STABILITY_HOURS,
            importance=5.0,
            priority=10
        )
        horizon_normal = engine.time_until_forgotten(state_normal)
        horizon_important = engine.time_until_forgotten(state_important)
        assert horizon_important > horizon_normal

    def test_spaced_repetition_pattern(self):
        """
        Review at expanding intervals — stability should keep growing.
        Simulates real spaced repetition learning.
        """
        engine = make_engine(fixed_clock(NOW))
        state = engine.create_state()
        review_gaps_hours = [1, 2, 4, 8, 16]  # expanding intervals
        stabilities = [state.stability_hours]

        current_time = NOW
        for gap in review_gaps_hours:
            current_time += hours_to_seconds(gap)
            engine = make_engine(fixed_clock(current_time))
            decayed = engine.decay(state)
            state = engine.reinforce(decayed, quality=1.0)
            stabilities.append(state.stability_hours)

        # stability should grow across all reviews
        assert stabilities[-1] > stabilities[0]