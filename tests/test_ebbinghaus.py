import math
import time
import pytest
from memory.ebbinghaus import (
    EbbinghausEngine,
    RetentionState,
    compute_retention,
    is_memory_forgotten,
    reinforce_memory,
    time_until_forgotten,
    create_memory_state,
    BASE_STABILITY_HOURS,
    MAX_STABILITY_HOURS,
    MIN_IMPORTANCE, MAX_IMPORTANCE,
    MIN_PRIORITY, MAX_PRIORITY,
    DEFAULT_RETENTION_THRESHOLD,
)

# ─── Helpers ───────────────────────────────────────────────

def fixed_clock(t):
    return lambda: t

NOW = 1_000_000.0  # fixed epoch seconds


# ─── RetentionState.create ─────────────────────────────────

class TestRetentionStateCreate:

    def test_defaults(self):
        s = RetentionState.create(clock=fixed_clock(NOW))
        assert s.stability_hours == BASE_STABILITY_HOURS
        assert s.retention == 1.0
        assert s.access_count == 0
        assert s.importance == 1.0
        assert s.priority == 1

    def test_stability_capped_at_max(self):
        s = RetentionState.create(stability_hours=99999, clock=fixed_clock(NOW))
        assert s.stability_hours == MAX_STABILITY_HOURS

    def test_invalid_stability_zero(self):
        with pytest.raises(ValueError, match="stability_hours must be positive"):
            RetentionState.create(stability_hours=0)

    def test_invalid_stability_negative(self):
        with pytest.raises(ValueError):
            RetentionState.create(stability_hours=-1)

    def test_importance_bounds(self):
        with pytest.raises(ValueError, match="importance"):
            RetentionState.create(importance=0.0)
        with pytest.raises(ValueError, match="importance"):
            RetentionState.create(importance=6.0)

    def test_priority_bounds(self):
        with pytest.raises(ValueError, match="priority"):
            RetentionState.create(priority=0)
        with pytest.raises(ValueError, match="priority"):
            RetentionState.create(priority=11)

    def test_immutable(self):
        s = RetentionState.create(clock=fixed_clock(NOW))
        with pytest.raises(Exception):
            s.retention = 0.5


# ─── EbbinghausEngine init ─────────────────────────────────

class TestEngineInit:

    def test_invalid_threshold_zero(self):
        with pytest.raises(ValueError):
            EbbinghausEngine(retention_threshold=0.0)

    def test_invalid_threshold_one(self):
        with pytest.raises(ValueError):
            EbbinghausEngine(retention_threshold=1.0)

    def test_invalid_boost(self):
        with pytest.raises(ValueError):
            EbbinghausEngine(reinforcement_boost=1.0)

    def test_invalid_diminishing(self):
        with pytest.raises(ValueError):
            EbbinghausEngine(diminishing_factor=-0.1)


# ─── compute_retention ─────────────────────────────────────

class TestComputeRetention:

    def setup_method(self):
        self.engine = EbbinghausEngine(clock=fixed_clock(NOW))

    def test_no_elapsed_time(self):
        r = self.engine.compute_retention(0.0, BASE_STABILITY_HOURS)
        assert r == pytest.approx(1.0)

    def test_one_stability_period(self):
        # At t=S, R = e^-1 ≈ 0.3679
        r = self.engine.compute_retention(BASE_STABILITY_HOURS, BASE_STABILITY_HOURS)
        assert r == pytest.approx(math.exp(-1), rel=1e-6)

    def test_large_elapsed_clamps_to_zero(self):
        r = self.engine.compute_retention(1e9, BASE_STABILITY_HOURS)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_negative_elapsed_treated_as_zero(self):
        r = self.engine.compute_retention(-10.0, BASE_STABILITY_HOURS)
        assert r == pytest.approx(1.0)

    def test_invalid_stability(self):
        with pytest.raises(ValueError):
            self.engine.compute_retention(1.0, 0.0)

    def test_stability_capped(self):
        r1 = self.engine.compute_retention(100.0, MAX_STABILITY_HOURS)
        r2 = self.engine.compute_retention(100.0, MAX_STABILITY_HOURS * 10)
        assert r1 == pytest.approx(r2)


# ─── decay ─────────────────────────────────────────────────

class TestDecay:

    def test_no_time_passed(self):
        engine = EbbinghausEngine(clock=fixed_clock(NOW))
        state = RetentionState.create(clock=fixed_clock(NOW))
        decayed = engine.decay(state)
        assert decayed.retention == pytest.approx(1.0)

    def test_retention_decreases_over_time(self):
        state = RetentionState.create(clock=fixed_clock(NOW))
        engine = EbbinghausEngine(clock=fixed_clock(NOW + 3600 * BASE_STABILITY_HOURS))
        decayed = engine.decay(state)
        assert decayed.retention == pytest.approx(math.exp(-1), rel=1e-6)

    def test_stability_unchanged_after_decay(self):
        state = RetentionState.create(clock=fixed_clock(NOW))
        engine = EbbinghausEngine(clock=fixed_clock(NOW + 3600))
        decayed = engine.decay(state)
        assert decayed.stability_hours == state.stability_hours

    def test_access_count_unchanged_after_decay(self):
        state = RetentionState.create(clock=fixed_clock(NOW))
        engine = EbbinghausEngine(clock=fixed_clock(NOW + 3600))
        decayed = engine.decay(state)
        assert decayed.access_count == state.access_count


# ─── reinforce ─────────────────────────────────────────────

class TestReinforce:

    def setup_method(self):
        self.engine = EbbinghausEngine(clock=fixed_clock(NOW))
        self.state = RetentionState.create(clock=fixed_clock(NOW))

    def test_retention_reset_to_one(self):
        reinforced = self.engine.reinforce(self.state)
        assert reinforced.retention == 1.0

    def test_access_count_incremented(self):
        reinforced = self.engine.reinforce(self.state)
        assert reinforced.access_count == self.state.access_count + 1

    def test_stability_increases(self):
        reinforced = self.engine.reinforce(self.state, quality=1.0)
        assert reinforced.stability_hours > self.state.stability_hours

    def test_quality_zero_minimal_boost(self):
        reinforced = self.engine.reinforce(self.state, quality=0.0)
        # effective_boost = 1.0, so stability unchanged
        assert reinforced.stability_hours == pytest.approx(self.state.stability_hours)

    def test_diminishing_returns(self):
        s = self.state
        boosts = []
        for _ in range(5):
            prev_stability = s.stability_hours
            s = self.engine.reinforce(s, quality=1.0)
            # measure the multiplier, not the absolute delta
            boosts.append(s.stability_hours / prev_stability)
        # Each successive multiplier should be smaller (closer to 1.0)
        assert all(boosts[i] > boosts[i+1] for i in range(len(boosts)-1))

    def test_stability_capped_at_max(self):
        big_state = RetentionState.create(
            stability_hours=MAX_STABILITY_HOURS, clock=fixed_clock(NOW)
        )
        reinforced = self.engine.reinforce(big_state, quality=1.0)
        assert reinforced.stability_hours <= MAX_STABILITY_HOURS

    def test_invalid_quality(self):
        with pytest.raises(ValueError):
            self.engine.reinforce(self.state, quality=1.1)
        with pytest.raises(ValueError):
            self.engine.reinforce(self.state, quality=-0.1)


# ─── forgotten ─────────────────────────────────────────────

class TestForgotten:

    def test_fresh_memory_not_forgotten(self):
        engine = EbbinghausEngine(clock=fixed_clock(NOW))
        state = RetentionState.create(clock=fixed_clock(NOW))
        assert engine.forgotten(state) is False

    def test_old_memory_is_forgotten(self):
        state = RetentionState.create(clock=fixed_clock(NOW))
        # Way in the future
        engine = EbbinghausEngine(clock=fixed_clock(NOW + 3600 * 10_000))
        assert engine.forgotten(state) is True


# ─── time_until_forgotten ──────────────────────────────────

class TestTimeUntilForgotten:

    def test_positive_remaining(self):
        engine = EbbinghausEngine(clock=fixed_clock(NOW))
        state = RetentionState.create(clock=fixed_clock(NOW))
        remaining = engine.time_until_forgotten(state)
        assert remaining > 0

    def test_already_forgotten_returns_zero(self):
        state = RetentionState.create(clock=fixed_clock(NOW))
        engine = EbbinghausEngine(clock=fixed_clock(NOW + 3600 * 100_000))
        assert engine.time_until_forgotten(state) == 0.0

    def test_formula_correctness(self):
        engine = EbbinghausEngine(
            retention_threshold=DEFAULT_RETENTION_THRESHOLD,
            clock=fixed_clock(NOW)
        )
        state = RetentionState.create(
            stability_hours=BASE_STABILITY_HOURS, clock=fixed_clock(NOW)
        )
        expected = -BASE_STABILITY_HOURS * math.log(DEFAULT_RETENTION_THRESHOLD)
        assert engine.time_until_forgotten(state) == pytest.approx(expected, rel=1e-4)


# ─── create_state (weighted stability) ────────────────────

class TestCreateState:

    def test_weighted_stability_applied(self):
        engine = EbbinghausEngine(clock=fixed_clock(NOW))
        state = engine.create_state(stability_hours=10.0, importance=2.0, priority=3)
        assert state.stability_hours == pytest.approx(min(10.0 * 2.0 * 3, MAX_STABILITY_HOURS))

    def test_weighted_stability_capped(self):
        engine = EbbinghausEngine(clock=fixed_clock(NOW))
        state = engine.create_state(stability_hours=10000.0, importance=5.0, priority=10)
        assert state.stability_hours == MAX_STABILITY_HOURS


# ─── Functional API ────────────────────────────────────────

class TestFunctionalAPI:

    def test_compute_retention_basic(self):
        now = time.time()
        r = compute_retention(now, BASE_STABILITY_HOURS, now=now)
        assert r == pytest.approx(1.0)

    def test_compute_retention_datetime(self):
        from datetime import datetime, timezone
        dt = datetime.now(timezone.utc)
        r = compute_retention(dt, BASE_STABILITY_HOURS, now=dt)
        assert r == pytest.approx(1.0)

    def test_is_memory_forgotten_above_threshold(self):
        assert is_memory_forgotten(DEFAULT_RETENTION_THRESHOLD + 0.01) is False

    def test_is_memory_forgotten_below_threshold(self):
        assert is_memory_forgotten(DEFAULT_RETENTION_THRESHOLD - 0.01) is True

    def test_reinforce_memory_increases_stability(self):
        new_s = reinforce_memory(BASE_STABILITY_HOURS, quality=1.0, review_count=0)
        assert new_s > BASE_STABILITY_HOURS

    def test_time_until_forgotten_positive(self):
        now = time.time()
        t = time_until_forgotten(now, BASE_STABILITY_HOURS, now=now)
        assert t > 0

    def test_create_memory_state_backward_compat(self):
        state = create_memory_state()
        assert isinstance(state, RetentionState)
        assert state.retention == 1.0