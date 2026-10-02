"""
Tests for WorkingMemory (working_memory.py)

Run with:
    pytest test_working_memory.py -v
"""

import copy
import threading
import pytest

from memory.working_memory import (
    WorkingMemory,
    WorkingMemoryItem,
    VALID_ROLES,
    DEFAULT_CAPACITY,
    MAX_CONTENT_LENGTH,
    TRUNCATION_SUFFIX,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_wm(capacity: int = 5) -> WorkingMemory:
    return WorkingMemory(capacity=capacity)


def fill(wm: WorkingMemory, n: int, role: str = "user"):
    for i in range(n):
        wm.add(role, f"message {i}")


# ─────────────────────────────────────────────────────────────
# WorkingMemoryItem dataclass
# ─────────────────────────────────────────────────────────────

class TestWorkingMemoryItem:

    def test_frozen_prevents_mutation(self):
        item = WorkingMemoryItem(role="user", content="hello")
        with pytest.raises((AttributeError, TypeError)):
            item.content = "changed"  # type: ignore[misc]

    def test_fields_stored(self):
        item = WorkingMemoryItem(role="assistant", content="hi")
        assert item.role == "assistant"
        assert item.content == "hi"


# ─────────────────────────────────────────────────────────────
# __init__
# ─────────────────────────────────────────────────────────────

class TestInit:

    def test_default_capacity(self):
        wm = WorkingMemory()
        assert wm.capacity == DEFAULT_CAPACITY

    def test_custom_capacity(self):
        wm = WorkingMemory(capacity=10)
        assert wm.capacity == 10

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError):
            WorkingMemory(capacity=0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError):
            WorkingMemory(capacity=-1)

    def test_starts_empty(self):
        wm = WorkingMemory()
        assert wm.is_empty()
        assert wm.size() == 0

    def test_metrics_zeroed_on_init(self):
        wm = WorkingMemory()
        m = wm.metrics()
        assert m["total_adds"] == 0
        assert m["total_evictions"] == 0
        assert m["total_snapshots"] == 0
        assert m["total_truncations"] == 0
        assert m["total_clears"] == 0


# ─────────────────────────────────────────────────────────────
# add()
# ─────────────────────────────────────────────────────────────

class TestAdd:

    def test_valid_roles_accepted(self):
        wm = make_wm()
        for role in VALID_ROLES:
            wm.add(role, "content")  # no exception

    def test_invalid_role_raises(self):
        wm = make_wm()
        with pytest.raises(ValueError, match="role"):
            wm.add("admin", "content")

    def test_empty_content_raises(self):
        wm = make_wm()
        with pytest.raises(ValueError, match="content"):
            wm.add("user", "")

    def test_whitespace_only_content_raises(self):
        wm = make_wm()
        with pytest.raises(ValueError):
            wm.add("user", "   ")

    def test_content_is_stripped(self):
        wm = make_wm()
        wm.add("user", "  hello  ")
        item = wm.recent(limit=1)[0]
        assert item.content == "hello"

    def test_role_is_stripped(self):
        wm = make_wm()
        wm.add("  user  ", "content")
        item = wm.recent(limit=1)[0]
        assert item.role == "user"

    def test_increments_add_counter(self):
        wm = make_wm()
        wm.add("user", "hello")
        assert wm.metrics()["total_adds"] == 1

    def test_size_increases(self):
        wm = make_wm()
        wm.add("user", "a")
        wm.add("assistant", "b")
        assert wm.size() == 2

    def test_truncates_long_content(self):
        wm = make_wm()
        long_content = "x" * (MAX_CONTENT_LENGTH + 100)
        wm.add("user", long_content)
        item = wm.recent(limit=1)[0]
        assert len(item.content) <= MAX_CONTENT_LENGTH
        assert item.content.endswith(TRUNCATION_SUFFIX)

    def test_truncation_increments_counter(self):
        wm = make_wm()
        wm.add("user", "x" * (MAX_CONTENT_LENGTH + 1))
        assert wm.metrics()["total_truncations"] == 1

    def test_content_at_exact_limit_not_truncated(self):
        wm = make_wm()
        content = "x" * MAX_CONTENT_LENGTH
        wm.add("user", content)
        item = wm.recent(limit=1)[0]
        assert not item.content.endswith(TRUNCATION_SUFFIX)
        assert wm.metrics()["total_truncations"] == 0

    def test_eviction_on_full_buffer(self):
        wm = make_wm(capacity=3)
        wm.add("user", "first")
        wm.add("user", "second")
        wm.add("user", "third")
        wm.add("user", "fourth")  # evicts "first"
        items = wm.recent(limit=10)
        contents = [i.content for i in items]
        assert "first" not in contents
        assert "fourth" in contents

    def test_eviction_increments_counter(self):
        wm = make_wm(capacity=2)
        wm.add("user", "a")
        wm.add("user", "b")
        wm.add("user", "c")  # evicts "a"
        assert wm.metrics()["total_evictions"] == 1

    def test_buffer_never_exceeds_capacity(self):
        wm = make_wm(capacity=3)
        fill(wm, 10)
        assert wm.size() == 3


# ─────────────────────────────────────────────────────────────
# recent()
# ─────────────────────────────────────────────────────────────

class TestRecent:

    def test_returns_list(self):
        wm = make_wm()
        fill(wm, 3)
        assert isinstance(wm.recent(), list)

    def test_returns_most_recent_n(self):
        wm = make_wm(capacity=10)
        fill(wm, 5)
        result = wm.recent(limit=3)
        assert len(result) == 3
        assert result[-1].content == "message 4"

    def test_limit_larger_than_buffer_returns_all(self):
        wm = make_wm(capacity=10)
        fill(wm, 3)
        assert len(wm.recent(limit=100)) == 3

    def test_zero_limit_raises(self):
        wm = make_wm()
        with pytest.raises(ValueError, match="limit"):
            wm.recent(limit=0)

    def test_negative_limit_raises(self):
        wm = make_wm()
        with pytest.raises(ValueError):
            wm.recent(limit=-1)

    def test_role_filter_user(self):
        wm = make_wm(capacity=10)
        wm.add("user", "user msg")
        wm.add("assistant", "assistant msg")
        wm.add("user", "user msg 2")
        result = wm.recent(limit=10, role_filter="user")
        assert all(i.role == "user" for i in result)
        assert len(result) == 2

    def test_role_filter_assistant(self):
        wm = make_wm(capacity=10)
        wm.add("user", "u")
        wm.add("assistant", "a1")
        wm.add("assistant", "a2")
        result = wm.recent(limit=10, role_filter="assistant")
        assert len(result) == 2

    def test_invalid_role_filter_raises(self):
        wm = make_wm()
        with pytest.raises(ValueError, match="role_filter"):
            wm.recent(role_filter="bot")

    def test_empty_memory_returns_empty_list(self):
        wm = make_wm()
        assert wm.recent() == []

    def test_result_is_snapshot_safe(self):
        wm = make_wm(capacity=10)
        fill(wm, 3)
        result = wm.recent(limit=10)
        wm.add("user", "new message")
        assert len(result) == 3  # snapshot unaffected


# ─────────────────────────────────────────────────────────────
# snapshot()
# ─────────────────────────────────────────────────────────────

class TestSnapshot:

    def test_returns_all_items(self):
        wm = make_wm(capacity=10)
        fill(wm, 4)
        snap = wm.snapshot()
        assert len(snap) == 4

    def test_snapshot_is_deep_copy(self):
        wm = make_wm(capacity=10)
        fill(wm, 3)
        snap = wm.snapshot()
        wm.clear()
        assert len(snap) == 3  # original snapshot intact

    def test_increments_snapshot_counter(self):
        wm = make_wm()
        wm.snapshot()
        wm.snapshot()
        assert wm.metrics()["total_snapshots"] == 2

    def test_empty_snapshot(self):
        wm = make_wm()
        assert wm.snapshot() == []

    def test_snapshot_order_preserved(self):
        wm = make_wm(capacity=10)
        wm.add("user", "first")
        wm.add("assistant", "second")
        wm.add("user", "third")
        snap = wm.snapshot()
        assert [i.content for i in snap] == ["first", "second", "third"]


# ─────────────────────────────────────────────────────────────
# as_messages()
# ─────────────────────────────────────────────────────────────

class TestAsMessages:

    def test_returns_list_of_dicts(self):
        wm = make_wm(capacity=10)
        fill(wm, 3)
        msgs = wm.as_messages()
        assert all(isinstance(m, dict) for m in msgs)

    def test_dict_has_role_and_content_keys(self):
        wm = make_wm()
        wm.add("user", "hello")
        msg = wm.as_messages()[0]
        assert "role" in msg
        assert "content" in msg

    def test_values_match_item(self):
        wm = make_wm()
        wm.add("assistant", "hi there")
        msg = wm.as_messages()[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "hi there"

    def test_limit_respected(self):
        wm = make_wm(capacity=10)
        fill(wm, 5)
        assert len(wm.as_messages(limit=2)) == 2

    def test_role_filter_applied(self):
        wm = make_wm(capacity=10)
        wm.add("user", "u")
        wm.add("assistant", "a")
        msgs = wm.as_messages(role_filter="user")
        assert all(m["role"] == "user" for m in msgs)

    def test_agent_id_accepted_without_error(self):
        wm = make_wm()
        wm.add("user", "hello")
        wm.as_messages(agent_id="agent-42")  # no exception

    def test_empty_memory_returns_empty_list(self):
        wm = make_wm()
        assert wm.as_messages() == []


# ─────────────────────────────────────────────────────────────
# format_for_prompt()
# ─────────────────────────────────────────────────────────────

class TestFormatForPrompt:

    def test_returns_string(self):
        wm = make_wm()
        wm.add("user", "hello")
        assert isinstance(wm.format_for_prompt(), str)

    def test_format_contains_role_and_content(self):
        wm = make_wm()
        wm.add("user", "hello world")
        result = wm.format_for_prompt()
        assert "user" in result
        assert "hello world" in result

    def test_entries_separated_by_newline(self):
        wm = make_wm(capacity=10)
        wm.add("user", "first")
        wm.add("assistant", "second")
        result = wm.format_for_prompt()
        assert "\n" in result

    def test_limit_respected(self):
        wm = make_wm(capacity=10)
        fill(wm, 5)
        result = wm.format_for_prompt(limit=2)
        assert result.count("\n") == 1  # 2 lines → 1 newline

    def test_role_filter_applied(self):
        wm = make_wm(capacity=10)
        wm.add("user", "user msg")
        wm.add("assistant", "assistant msg")
        result = wm.format_for_prompt(role_filter="user")
        assert "assistant" not in result

    def test_empty_memory_returns_empty_string(self):
        wm = make_wm()
        assert wm.format_for_prompt() == ""


# ─────────────────────────────────────────────────────────────
# clear()
# ─────────────────────────────────────────────────────────────

class TestClear:

    def test_empties_buffer(self):
        wm = make_wm()
        fill(wm, 3)
        wm.clear()
        assert wm.is_empty()
        assert wm.size() == 0

    def test_increments_clear_counter(self):
        wm = make_wm()
        wm.clear()
        wm.clear()
        assert wm.metrics()["total_clears"] == 2

    def test_can_add_after_clear(self):
        wm = make_wm()
        fill(wm, 3)
        wm.clear()
        wm.add("user", "fresh start")
        assert wm.size() == 1


# ─────────────────────────────────────────────────────────────
# size() and is_empty()
# ─────────────────────────────────────────────────────────────

class TestSizeAndIsEmpty:

    def test_empty_initially(self):
        wm = make_wm()
        assert wm.size() == 0
        assert wm.is_empty() is True

    def test_not_empty_after_add(self):
        wm = make_wm()
        wm.add("user", "hi")
        assert wm.is_empty() is False

    def test_size_tracks_adds(self):
        wm = make_wm(capacity=10)
        for i in range(7):
            wm.add("user", f"msg {i}")
        assert wm.size() == 7

    def test_len_matches_size(self):
        wm = make_wm(capacity=10)
        fill(wm, 4)
        assert len(wm) == wm.size() == 4


# ─────────────────────────────────────────────────────────────
# metrics()
# ─────────────────────────────────────────────────────────────

class TestMetrics:

    def test_returns_dict(self):
        wm = make_wm()
        assert isinstance(wm.metrics(), dict)

    def test_capacity_key_present(self):
        wm = make_wm(capacity=7)
        assert wm.metrics()["capacity"] == 7

    def test_current_size_reflects_buffer(self):
        wm = make_wm(capacity=10)
        fill(wm, 4)
        assert wm.metrics()["current_size"] == 4

    def test_all_expected_keys_present(self):
        wm = make_wm()
        m = wm.metrics()
        for key in (
            "capacity", "current_size", "total_adds",
            "total_evictions", "total_snapshots",
            "total_truncations", "total_clears",
        ):
            assert key in m, f"Missing key: {key}"

    def test_combined_operations(self):
        wm = make_wm(capacity=3)
        fill(wm, 5)              # 2 evictions, 5 adds
        wm.snapshot()            # 1 snapshot
        wm.clear()               # 1 clear
        wm.add("user", "x" * (MAX_CONTENT_LENGTH + 1))  # 1 truncation
        m = wm.metrics()
        assert m["total_adds"] == 6
        assert m["total_evictions"] == 2
        assert m["total_snapshots"] == 1
        assert m["total_clears"] == 1
        assert m["total_truncations"] == 1


# ─────────────────────────────────────────────────────────────
# Thread safety
# ─────────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_adds(self):
        wm = WorkingMemory(capacity=200)
        errors = []

        def add_items(thread_id):
            try:
                for i in range(10):
                    wm.add("user", f"thread {thread_id} msg {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_items, args=(t,))
            for t in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert wm.metrics()["total_adds"] == 100

    def test_concurrent_reads_and_writes(self):
        wm = WorkingMemory(capacity=50)
        fill(wm, 10)
        errors = []

        def reader():
            try:
                wm.recent(limit=5)
                wm.snapshot()
            except Exception as e:
                errors.append(e)

        def writer(i):
            try:
                wm.add("assistant", f"write {i}")
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=reader) for _ in range(10)]
            + [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_clear_and_add(self):
        wm = WorkingMemory(capacity=50)
        errors = []

        def adder():
            try:
                for _ in range(5):
                    wm.add("user", "content")
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                for _ in range(3):
                    wm.clear()
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=adder) for _ in range(5)]
            + [threading.Thread(target=clearer) for _ in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"