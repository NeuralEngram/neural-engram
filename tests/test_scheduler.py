# tests/test_scheduler.py

import time
import threading
import pytest
from unittest.mock import MagicMock, patch, call
from memory.scheduler import (
    MemoryScheduler,
    CleanupResult,
    MaintenanceEvent,
    FAILURE_ALERT_THRESHOLD,
)


# ─── Fixtures ──────────────────────────────────────────────

def make_store(cleanup_return=0):
    store = MagicMock()
    store.apply_decay.return_value = None
    store.cleanup.return_value = cleanup_return
    store.metrics.return_value = {"count": 10}
    return store

def make_scheduler(interval=300, store=None, on_cycle=None, on_failure=None):
    if store is None:
        store = make_store()
    return MemoryScheduler(
        memory_store=store,
        interval_seconds=interval,
        on_cycle_complete=on_cycle,
        on_failure_alert=on_failure
    )


# ═══════════════════════════════════════════════════════════
# 1. INIT & VALIDATION
# ═══════════════════════════════════════════════════════════

class TestInit:

    def test_valid_interval_accepted(self):
        s = make_scheduler(interval=60)
        assert s.interval_seconds == 60

    def test_zero_interval_raises(self):
        with pytest.raises(ValueError, match="interval_seconds must be > 0"):
            make_scheduler(interval=0)

    def test_negative_interval_raises(self):
        with pytest.raises(ValueError):
            make_scheduler(interval=-1)

    def test_initial_state(self):
        s = make_scheduler()
        assert s.running is False
        assert s.total_cycles == 0
        assert s.total_failures == 0
        assert s.total_removed == 0
        assert s.consecutive_failures == 0
        assert s.last_cycle_time is None
        assert s.thread is None

    def test_callbacks_stored(self):
        on_cycle = MagicMock()
        on_failure = MagicMock()
        s = make_scheduler(on_cycle=on_cycle, on_failure=on_failure)
        assert s.on_cycle_complete is on_cycle
        assert s.on_failure_alert is on_failure

    def test_no_callbacks_accepted(self):
        s = make_scheduler()
        assert s.on_cycle_complete is None
        assert s.on_failure_alert is None


# ═══════════════════════════════════════════════════════════
# 2. CleanupResult & MaintenanceEvent
# ═══════════════════════════════════════════════════════════

class TestCleanupResult:

    def test_fields(self):
        r = CleanupResult(removed_count=3, removed_ids=["a", "b", "c"])
        assert r.removed_count == 3
        assert r.removed_ids == ["a", "b", "c"]

    def test_immutable(self):
        r = CleanupResult(removed_count=1, removed_ids=[])
        with pytest.raises(Exception):
            r.removed_count = 5

    def test_zero_removed(self):
        r = CleanupResult(removed_count=0, removed_ids=[])
        assert r.removed_count == 0
        assert r.removed_ids == []


class TestMaintenanceEvent:

    def test_immutable(self):
        event = MaintenanceEvent(
            cycle_id="c1", removed_count=0, removed_ids=[],
            duration=0.1, average_duration=0.1, timestamp=time.time(),
            total_cycles=1, consecutive_failures=0,
            scheduler_metrics={}, memory_store_metrics={}, memory_snapshot={}
        )
        with pytest.raises(Exception):
            event.removed_count = 5

    def test_all_fields_accessible(self):
        now = time.time()
        event = MaintenanceEvent(
            cycle_id="abc", removed_count=2, removed_ids=["x"],
            duration=0.5, average_duration=0.5, timestamp=now,
            total_cycles=3, consecutive_failures=0,
            scheduler_metrics={"k": 1}, memory_store_metrics={}, memory_snapshot={}
        )
        assert event.cycle_id == "abc"
        assert event.removed_count == 2
        assert event.removed_ids == ["x"]
        assert event.total_cycles == 3


# ═══════════════════════════════════════════════════════════
# 3. run_once() / _maintenance_cycle()
# ═══════════════════════════════════════════════════════════

class TestRunOnce:

    def test_calls_apply_decay(self):
        store = make_store()
        s = make_scheduler(store=store)
        s.run_once()
        store.apply_decay.assert_called_once()

    def test_calls_cleanup(self):
        store = make_store()
        s = make_scheduler(store=store)
        s.run_once()
        store.cleanup.assert_called_once()

    def test_increments_total_cycles(self):
        s = make_scheduler()
        s.run_once()
        assert s.total_cycles == 1

    def test_total_cycles_accumulates(self):
        s = make_scheduler()
        s.run_once()
        s.run_once()
        assert s.total_cycles == 2

    def test_total_removed_updated(self):
        store = make_store(cleanup_return=5)
        s = make_scheduler(store=store)
        s.run_once()
        assert s.total_removed == 5

    def test_total_removed_accumulates(self):
        store = make_store(cleanup_return=3)
        s = make_scheduler(store=store)
        s.run_once()
        s.run_once()
        assert s.total_removed == 6

    def test_last_cycle_time_set(self):
        s = make_scheduler()
        before = time.time()
        s.run_once()
        assert s.last_cycle_time >= before

    def test_consecutive_failures_reset_on_success(self):
        s = make_scheduler()
        s.consecutive_failures = 3
        s.run_once()
        assert s.consecutive_failures == 0

    def test_integer_cleanup_backward_compat(self):
        """cleanup() returning int should be wrapped in CleanupResult."""
        store = make_store(cleanup_return=7)
        s = make_scheduler(store=store)
        s.run_once()
        assert s.total_removed == 7

    def test_cleanup_result_object_accepted(self):
        store = make_store()
        store.cleanup.return_value = CleanupResult(
            removed_count=4, removed_ids=["a", "b", "c", "d"]
        )
        s = make_scheduler(store=store)
        s.run_once()
        assert s.total_removed == 4

    def test_on_cycle_complete_called(self):
        on_cycle = MagicMock()
        s = make_scheduler(on_cycle=on_cycle)
        s.run_once()
        on_cycle.assert_called_once()
        event = on_cycle.call_args[0][0]
        assert isinstance(event, MaintenanceEvent)

    def test_on_cycle_complete_event_fields(self):
        on_cycle = MagicMock()
        store = make_store(cleanup_return=2)
        s = make_scheduler(store=store, on_cycle=on_cycle)
        s.run_once()
        event = on_cycle.call_args[0][0]
        assert event.removed_count == 2
        assert event.total_cycles == 1
        assert event.consecutive_failures == 0
        assert event.duration >= 0

    def test_cycle_id_unique_per_run(self):
        on_cycle = MagicMock()
        s = make_scheduler(on_cycle=on_cycle)
        s.run_once()
        s.run_once()
        ids = [call[0][0].cycle_id for call in on_cycle.call_args_list]
        assert ids[0] != ids[1]

    def test_callback_exception_does_not_crash_cycle(self):
        on_cycle = MagicMock(side_effect=Exception("callback error"))
        s = make_scheduler(on_cycle=on_cycle)
        s.run_once()  # should not raise
        assert s.total_cycles == 1


# ═══════════════════════════════════════════════════════════
# 4. FAILURE HANDLING
# ═══════════════════════════════════════════════════════════

class TestFailureHandling:

    def test_failure_increments_total_failures(self):
        store = make_store()
        store.apply_decay.side_effect = Exception("decay error")
        s = make_scheduler(store=store)
        s.run_once()
        assert s.total_failures == 1

    def test_failure_increments_consecutive_failures(self):
        store = make_store()
        store.apply_decay.side_effect = Exception("error")
        s = make_scheduler(store=store)
        s.run_once()
        assert s.consecutive_failures == 1

    def test_consecutive_failures_accumulate(self):
        store = make_store()
        store.apply_decay.side_effect = Exception("error")
        s = make_scheduler(store=store)
        s.run_once()
        s.run_once()
        assert s.consecutive_failures == 2

    def test_success_resets_consecutive_failures(self):
        store = make_store()
        store.apply_decay.side_effect = Exception("error")
        s = make_scheduler(store=store)
        s.run_once()
        assert s.consecutive_failures == 1
        store.apply_decay.side_effect = None
        s.run_once()
        assert s.consecutive_failures == 0

    def test_failure_does_not_increment_total_cycles(self):
        store = make_store()
        store.apply_decay.side_effect = Exception("error")
        s = make_scheduler(store=store)
        s.run_once()
        assert s.total_cycles == 0

    def test_alert_triggered_at_threshold(self):
        on_failure = MagicMock()
        store = make_store()
        store.apply_decay.side_effect = Exception("error")
        s = make_scheduler(store=store, on_failure=on_failure)
        for _ in range(FAILURE_ALERT_THRESHOLD):
            s.run_once()
        on_failure.assert_called_once()

    def test_alert_not_triggered_below_threshold(self):
        on_failure = MagicMock()
        store = make_store()
        store.apply_decay.side_effect = Exception("error")
        s = make_scheduler(store=store, on_failure=on_failure)
        for _ in range(FAILURE_ALERT_THRESHOLD - 1):
            s.run_once()
        on_failure.assert_not_called()

    def test_alert_payload_fields(self):
        on_failure = MagicMock()
        store = make_store()
        store.apply_decay.side_effect = Exception("error")
        s = make_scheduler(store=store, on_failure=on_failure)
        for _ in range(FAILURE_ALERT_THRESHOLD):
            s.run_once()
        payload = on_failure.call_args[0][0]
        assert "cycle_id" in payload
        assert "consecutive_failures" in payload
        assert "total_failures" in payload
        assert "scheduler_metrics" in payload
        assert "system_health" in payload
        assert "timestamp" in payload

    def test_failure_alert_callback_exception_does_not_crash(self):
        on_failure = MagicMock(side_effect=Exception("alert error"))
        store = make_store()
        store.apply_decay.side_effect = Exception("cycle error")
        s = make_scheduler(store=store, on_failure=on_failure)
        for _ in range(FAILURE_ALERT_THRESHOLD):
            s.run_once()  # should not raise


# ═══════════════════════════════════════════════════════════
# 5. BACKOFF
# ═══════════════════════════════════════════════════════════

class TestComputeBackoff:

    def test_no_failures_returns_interval(self):
        s = make_scheduler(interval=60)
        assert s._compute_backoff() == 60

    def test_one_failure_doubles_interval(self):
        s = make_scheduler(interval=60)
        s.consecutive_failures = 1
        assert s._compute_backoff() == 120

    def test_two_failures_quadruples(self):
        s = make_scheduler(interval=60)
        s.consecutive_failures = 2
        assert s._compute_backoff() == 240

    def test_backoff_capped_at_32x(self):
        s = make_scheduler(interval=60)
        s.consecutive_failures = 100
        assert s._compute_backoff() == 60 * 32

    def test_backoff_resets_after_recovery(self):
        s = make_scheduler(interval=60)
        s.consecutive_failures = 3
        assert s._compute_backoff() > 60
        s.consecutive_failures = 0
        assert s._compute_backoff() == 60


# ═══════════════════════════════════════════════════════════
# 6. METRICS
# ═══════════════════════════════════════════════════════════

class TestMetrics:

    def test_metrics_returns_dict(self):
        s = make_scheduler()
        result = s.metrics()
        assert isinstance(result, dict)

    def test_metrics_keys(self):
        s = make_scheduler()
        m = s.metrics()
        expected = {
            "running", "interval_seconds", "total_cycles",
            "total_failures", "consecutive_failures", "total_removed",
            "last_cycle_time", "average_cycle_duration", "max_cycle_duration"
        }
        assert expected.issubset(m.keys())

    def test_metrics_initial_values(self):
        s = make_scheduler(interval=120)
        m = s.metrics()
        assert m["running"] is False
        assert m["interval_seconds"] == 120
        assert m["total_cycles"] == 0
        assert m["total_failures"] == 0
        assert m["total_removed"] == 0
        assert m["last_cycle_time"] is None

    def test_metrics_updated_after_run_once(self):
        store = make_store(cleanup_return=3)
        s = make_scheduler(store=store)
        s.run_once()
        m = s.metrics()
        assert m["total_cycles"] == 1
        assert m["total_removed"] == 3
        assert m["last_cycle_time"] is not None

    def test_average_duration_zero_before_any_cycle(self):
        s = make_scheduler()
        assert s.metrics()["average_cycle_duration"] == 0.0

    def test_average_duration_after_cycles(self):
        s = make_scheduler()
        s.run_once()
        # manually set a known duration so the test isn't timing-dependent
        s.total_cycle_duration = 1.0
        assert s._average_duration() == pytest.approx(1.0)

    def test_max_cycle_duration_tracked(self):
        s = make_scheduler()
        s.run_once()
        assert s.metrics()["max_cycle_duration"] >= 0.0


# ═══════════════════════════════════════════════════════════
# 7. AVERAGE DURATION HELPERS
# ═══════════════════════════════════════════════════════════

class TestAverageDuration:

    def test_zero_before_any_cycle(self):
        s = make_scheduler()
        assert s._average_duration() == 0.0

    def test_correct_after_known_durations(self):
        s = make_scheduler()
        s.total_cycles = 4
        s.total_cycle_duration = 2.0
        assert s._average_duration() == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════
# 8. SAFE HELPERS
# ═══════════════════════════════════════════════════════════

class TestSafeHelpers:

    def test_safe_memory_metrics_returns_dict(self):
        s = make_scheduler()
        result = s._safe_memory_metrics()
        assert isinstance(result, dict)

    def test_safe_memory_metrics_on_exception(self):
        store = make_store()
        store.metrics.side_effect = Exception("metrics error")
        s = make_scheduler(store=store)
        result = s._safe_memory_metrics()
        assert result == {}

    def test_safe_memory_snapshot_returns_dict(self):
        s = make_scheduler()
        result = s._safe_memory_snapshot()
        assert isinstance(result, dict)

    def test_safe_memory_snapshot_with_working_memory(self):
        store = make_store()
        store.working_memory = ["msg1", "msg2"]
        s = make_scheduler(store=store)
        result = s._safe_memory_snapshot()
        assert result.get("working_memory_size") == 2

    def test_safe_memory_snapshot_exception_returns_empty(self):
        store = make_store()
        # make hasattr return True but len() fail
        type(store).working_memory = property(lambda self: (_ for _ in ()).throw(Exception("snap error")))
        s = make_scheduler(store=store)
        result = s._safe_memory_snapshot()
        assert isinstance(result, dict)

    def test_system_health_returns_dict(self):
        s = make_scheduler()
        result = s._system_health()
        assert "thread_alive" in result
        assert "scheduler_running" in result
        assert "active_threads" in result

    def test_system_health_not_running(self):
        s = make_scheduler()
        health = s._system_health()
        assert health["thread_alive"] is False
        assert health["scheduler_running"] is False


# ═══════════════════════════════════════════════════════════
# 9. START / STOP / is_running
# ═══════════════════════════════════════════════════════════

class TestStartStop:

    def test_is_running_false_before_start(self):
        s = make_scheduler()
        assert s.is_running() is False

    def test_start_sets_running(self):
        s = make_scheduler()
        s.start()
        try:
            assert s.is_running() is True
        finally:
            s.stop()

    def test_stop_clears_running(self):
        s = make_scheduler()
        s.start()
        s.stop()
        assert s.is_running() is False

    def test_double_start_idempotent(self):
        s = make_scheduler()
        s.start()
        s.start()  # should not raise or create second thread
        try:
            assert s.is_running() is True
        finally:
            s.stop()

    def test_stop_before_start_no_error(self):
        s = make_scheduler()
        s.stop()  # should not raise

    def test_thread_is_daemon(self):
        s = make_scheduler()
        s.start()
        try:
            assert s.thread.daemon is True
        finally:
            s.stop()

    def test_thread_name(self):
        s = make_scheduler()
        s.start()
        try:
            assert s.thread.name == "MemoryScheduler"
        finally:
            s.stop()

    def test_start_triggers_immediate_cycle(self):
        store = make_store()
        s = make_scheduler(store=store, interval=300)
        s.start()
        time.sleep(0.2)
        try:
            assert store.apply_decay.call_count >= 1
        finally:
            s.stop()

    def test_stop_joins_thread(self):
        s = make_scheduler()
        s.start()
        s.stop()
        assert not s.thread.is_alive()


# ═══════════════════════════════════════════════════════════
# 10. THREAD SAFETY
# ═══════════════════════════════════════════════════════════

class TestThreadSafety:

    def test_concurrent_run_once_no_exception(self):
        s = make_scheduler()
        errors = []

        def worker():
            try:
                s.run_once()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_metrics_consistent_under_concurrent_cycles(self):
        store = make_store(cleanup_return=1)
        s = make_scheduler(store=store)
        threads = [threading.Thread(target=s.run_once) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert s.total_cycles == 20
        assert s.total_removed == 20
