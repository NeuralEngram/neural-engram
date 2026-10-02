# tests/test_scheduler_failures.py
"""
Scheduler failure escalation with a REAL MemoryScheduler over the REAL
memory stack. Failures are forced by making the real episodic store's
apply_decay()/prune_forgotten() raise.

Documented behaviour (memory/scheduler.py):
  - each failed cycle: total_failures += 1, consecutive_failures += 1
  - consecutive_failures >= FAILURE_ALERT_THRESHOLD → on_failure_alert
  - a successful cycle resets consecutive_failures to 0
  - backoff = interval * 2**consecutive_failures, capped at 32x
"""

import threading
import time

import pytest

from memory.scheduler import FAILURE_ALERT_THRESHOLD, MemoryScheduler


class Breaker:
    """Turns a real method into one that fails while `on` is True."""

    def __init__(self, obj, name, monkeypatch):
        self.on = True
        self.calls = 0
        real = getattr(obj, name)

        def wrapper(*a, **k):
            self.calls += 1
            if self.on:
                raise RuntimeError(f"forced {name} failure")
            return real(*a, **k)
        monkeypatch.setattr(obj, name, wrapper)


@pytest.fixture
def sched_parts(real_memory):
    alerts, cycles = [], []
    sched = MemoryScheduler(
        real_memory.store,
        interval_seconds=10,
        on_cycle_complete=cycles.append,
        on_failure_alert=alerts.append,
    )
    return sched, alerts, cycles


@pytest.mark.parametrize("method", ["apply_decay", "prune_forgotten"])
class TestEscalation:

    def test_no_alert_below_threshold(self, real_memory, sched_parts,
                                      monkeypatch, method):
        sched, alerts, _ = sched_parts
        Breaker(real_memory.episodic, method, monkeypatch)

        for _ in range(FAILURE_ALERT_THRESHOLD - 1):
            sched.run_once()

        assert alerts == []
        assert sched.consecutive_failures == FAILURE_ALERT_THRESHOLD - 1

    def test_alert_fires_at_threshold(self, real_memory, sched_parts,
                                      monkeypatch, method):
        sched, alerts, cycles = sched_parts
        Breaker(real_memory.episodic, method, monkeypatch)

        for _ in range(FAILURE_ALERT_THRESHOLD):
            sched.run_once()

        assert len(alerts) == 1
        a = alerts[0]
        assert a["consecutive_failures"] == FAILURE_ALERT_THRESHOLD
        assert a["total_failures"] == FAILURE_ALERT_THRESHOLD
        for key in ("cycle_id", "scheduler_metrics", "memory_store_metrics",
                    "memory_snapshot", "system_health", "timestamp"):
            assert key in a
        assert cycles == []  # failed cycles never reach on_cycle_complete

    def test_alert_repeats_every_failure_past_threshold(
        self, real_memory, sched_parts, monkeypatch, method
    ):
        """Characterisation: not once — every failed cycle at/after N."""
        sched, alerts, _ = sched_parts
        Breaker(real_memory.episodic, method, monkeypatch)

        for _ in range(FAILURE_ALERT_THRESHOLD + 3):
            sched.run_once()

        assert [a["consecutive_failures"] for a in alerts] == \
            list(range(FAILURE_ALERT_THRESHOLD, FAILURE_ALERT_THRESHOLD + 4))

    def test_success_resets_consecutive_not_total(
        self, real_memory, sched_parts, monkeypatch, method
    ):
        sched, alerts, cycles = sched_parts
        brk = Breaker(real_memory.episodic, method, monkeypatch)

        for _ in range(FAILURE_ALERT_THRESHOLD):
            sched.run_once()
        brk.on = False
        sched.run_once()

        assert sched.consecutive_failures == 0
        assert sched.total_failures == FAILURE_ALERT_THRESHOLD
        assert len(cycles) == 1
        assert cycles[0].consecutive_failures == 0

        # needs a FULL new streak before alerting again
        brk.on = True
        for _ in range(FAILURE_ALERT_THRESHOLD - 1):
            sched.run_once()
        assert len(alerts) == 1
        sched.run_once()
        assert len(alerts) == 2
        assert alerts[-1]["consecutive_failures"] == FAILURE_ALERT_THRESHOLD
        assert alerts[-1]["total_failures"] == 2 * FAILURE_ALERT_THRESHOLD


class TestResilience:

    def test_broken_alert_callback_does_not_break_scheduler(
        self, real_memory, monkeypatch
    ):
        def explode(payload):
            raise RuntimeError("pager is down")

        sched = MemoryScheduler(real_memory.store, interval_seconds=10,
                                on_failure_alert=explode)
        brk = Breaker(real_memory.episodic, "apply_decay", monkeypatch)

        for _ in range(FAILURE_ALERT_THRESHOLD + 1):
            sched.run_once()  # must not raise
        brk.on = False
        sched.run_once()

        assert sched.consecutive_failures == 0

    def test_alert_payload_survives_broken_store(
        self, real_memory, sched_parts, monkeypatch
    ):
        """Metrics/snapshot in the payload are 'safe' even if they fail."""
        sched, alerts, _ = sched_parts
        Breaker(real_memory.episodic, "apply_decay", monkeypatch)
        Breaker(real_memory.store, "metrics", monkeypatch)

        for _ in range(FAILURE_ALERT_THRESHOLD):
            sched.run_once()

        assert len(alerts) == 1

    def test_backoff_doubles_caps_and_resets(self, real_memory, monkeypatch):
        sched = MemoryScheduler(real_memory.store, interval_seconds=10)
        brk = Breaker(real_memory.episodic, "apply_decay", monkeypatch)

        seen = []
        for _ in range(8):
            sched.run_once()
            seen.append(sched._compute_backoff())
        assert seen == [20, 40, 80, 160, 320, 320, 320, 320]

        brk.on = False
        sched.run_once()
        assert sched._compute_backoff() == 10

    def test_background_loop_escalates_and_keeps_running(
        self, real_memory, monkeypatch
    ):
        """The real thread, not run_once(): alert must fire from the loop."""
        fired = threading.Event()
        sched = MemoryScheduler(real_memory.store, interval_seconds=0.0005,
                                on_failure_alert=lambda p: fired.set())
        brk = Breaker(real_memory.episodic, "apply_decay", monkeypatch)

        sched.start()
        try:
            assert fired.wait(timeout=20), "alert never fired from loop"
            assert sched.is_running()
            brk.on = False
            deadline = time.time() + 20
            while sched.consecutive_failures and time.time() < deadline:
                time.sleep(0.01)
            assert sched.consecutive_failures == 0
        finally:
            sched.stop()
        assert not sched.is_running()


class TestOrchestratorWiring:

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Needs a decision: Orchestrator builds MemoryScheduler with "
            "no on_failure_alert, so in the live app escalation only "
            "reaches logger.error. And because failed cycles skip "
            "on_cycle_complete, caregiver alerts also stop for the whole "
            "failure streak — silently."
        ),
    )
    def test_live_scheduler_has_failure_alert_wired(self, real_orchestrator):
        orch = real_orchestrator()
        assert orch.scheduler.on_failure_alert is not None

    def test_failure_streak_suspends_caregiver_alerts(
        self, real_orchestrator, monkeypatch
    ):
        """Characterisation of the consequence above."""
        from datetime import timedelta

        orch = real_orchestrator()
        orch.route("my daughter brings my medicine", agent_id="rajan")
        mo = orch.memory_store.orchestrator
        for ep in mo.episodic_memory.get_all():
            ep.last_reviewed_at -= timedelta(hours=40)  # warning band
        brk = Breaker(mo.episodic_memory, "apply_decay", monkeypatch)

        for _ in range(FAILURE_ALERT_THRESHOLD + 2):
            orch.scheduler.run_once()

        assert orch.notifier._already_notified == set()

        # control: the same memory DOES alert as soon as a cycle succeeds
        brk.on = False
        orch.scheduler.run_once()
        assert orch.notifier._already_notified
