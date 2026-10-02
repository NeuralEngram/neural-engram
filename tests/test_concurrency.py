# tests/test_concurrency.py
"""
Concurrency under real load, on real objects.

Every scenario:
  - releases all threads at once through a Barrier (maximises overlap)
  - treats a thread still alive after DEADLOCK_TIMEOUT as a deadlock and
    dumps every thread's stack so the lock cycle is visible
  - afterwards cross-checks all three tiers + the SQLite file for
    corruption: lost writes, orphans, duplicates, mis-attributed agents
"""

import faulthandler
import sqlite3
import sys
import threading
import time
from datetime import timedelta

import pytest

import memory.episodic_memory as episodic_module
from memory.scheduler import MemoryScheduler
from tests.conftest import hash_embed


DEADLOCK_TIMEOUT = 60
THREADS = 8
OPS = 25


def run_concurrently(workers):
    """Run callables in parallel; return exceptions; fail on deadlock."""
    barrier = threading.Barrier(len(workers))
    errors = []

    def wrap(fn):
        def target():
            try:
                barrier.wait()
                fn()
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        return target

    threads = [threading.Thread(target=wrap(fn), daemon=True)
               for fn in workers]
    for t in threads:
        t.start()
    deadline = time.time() + DEADLOCK_TIMEOUT
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.time()))
    alive = [t for t in threads if t.is_alive()]
    if alive:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        pytest.fail(f"deadlock: {len(alive)} thread(s) never finished")
    return errors


def db_ids(real_memory):
    conn = sqlite3.connect(real_memory.db_path)
    try:
        return {r[0] for r in conn.execute("SELECT episode_id FROM episodes")}
    finally:
        conn.close()


def assert_tiers_consistent(real_memory):
    """RAM == SQLite == Qdrant, one entry per id, same content/owner."""
    ram = {ep.episode_id: ep for ep in real_memory.episodic.get_all()}
    assert len(ram) == len(real_memory.episodic.get_all()), "duplicate ids"
    assert set(ram) == db_ids(real_memory), "RAM and SQLite diverged"
    assert real_memory.semantic.count() == len(ram), "Qdrant diverged"
    for mid, ep in ram.items():
        sem = real_memory.semantic.get(mid)
        assert sem is not None, f"missing in Qdrant: {mid}"
        assert sem.content == ep.content
        assert sem.metadata["agent_id"] == ep.agent_id
    return ram


def content_for(agent, t, n):
    return f"{agent} note thread{t} item{n} marker{agent}"


def owner_of(content):
    return content.split(" ", 1)[0]


# ─── store() ───────────────────────────────────────────────

class TestConcurrentStore:

    def test_same_agent_no_lost_or_duplicated_writes(self, real_memory):
        def worker(t):
            def fn():
                for n in range(OPS):
                    c = content_for("rajan", t, n)
                    real_memory.store.store("rajan", c, hash_embed(c))
            return fn

        errors = run_concurrently([worker(t) for t in range(THREADS)])

        assert errors == []
        ram = assert_tiers_consistent(real_memory)
        expected = {content_for("rajan", t, n)
                    for t in range(THREADS) for n in range(OPS)}
        assert {ep.content for ep in ram.values()} == expected
        wm = real_memory.working.as_messages(agent_id="rajan", limit=50)
        assert len(wm) == real_memory.working.capacity
        assert {m["content"] for m in wm} <= expected

    def test_different_agents_never_misattributed(self, real_memory):
        agents = [f"agent{i}" for i in range(THREADS)]

        def worker(agent):
            def fn():
                for n in range(OPS):
                    c = content_for(agent, 0, n)
                    real_memory.store.store(agent, c, hash_embed(c))
            return fn

        errors = run_concurrently([worker(a) for a in agents])

        assert errors == []
        ram = assert_tiers_consistent(real_memory)
        assert len(ram) == THREADS * OPS
        for ep in ram.values():
            assert owner_of(ep.content) == ep.agent_id
        for agent in agents:
            wm = real_memory.working.as_messages(agent_id=agent, limit=50)
            assert len(wm) == min(OPS, real_memory.working.capacity)
            assert all(owner_of(m["content"]) == agent for m in wm)


# ─── store() + retrieve() together ─────────────────────────

class TestConcurrentStoreAndRetrieve:

    @pytest.mark.parametrize("readers_share_agent", [True, False])
    def test_readers_only_ever_see_own_agent(
        self, real_memory, readers_share_agent
    ):
        writers = ["rajan", "alice"]
        violations = []

        def writer(agent, t):
            def fn():
                for n in range(OPS):
                    c = content_for(agent, t, n)
                    real_memory.store.store(agent, c, hash_embed(c))
            return fn

        def reader(agent):
            def fn():
                q = f"note marker{agent}"
                for _ in range(OPS):
                    r = real_memory.store.retrieve(
                        agent_id=agent, query=q, embedding=hash_embed(q))
                    for tier in ("episodic_memory", "semantic_memory",
                                 "fused_memory"):
                        for m in r[tier]:
                            if owner_of(m.content) != agent:
                                violations.append((agent, tier, m.content))
                    for m in r["working_memory"]:
                        if owner_of(m["content"]) != agent:
                            violations.append((agent, "wm", m["content"]))
            return fn

        workers = [writer(a, t) for a in writers for t in range(3)]
        reader_agents = writers if readers_share_agent else ["bob", "carol"]
        workers += [reader(a) for a in reader_agents for _ in range(3)]

        errors = run_concurrently(workers)

        assert errors == []
        assert violations == []
        ram = assert_tiers_consistent(real_memory)
        assert len(ram) == len(writers) * 3 * OPS


# ─── Read-modify-write on ONE memory ───────────────────────

class TestNoLostUpdates:

    def test_parallel_reinforce_counts_every_call(self, real_memory):
        mid = real_memory.store.store(
            "rajan", "shared target", hash_embed("shared target"))["memory_id"]

        def fn():
            for _ in range(OPS):
                real_memory.store.reinforce(agent_id="rajan", memory_id=mid)

        errors = run_concurrently([fn] * THREADS)

        assert errors == []
        total = THREADS * OPS
        assert real_memory.episodic.get_by_id(mid).review_count == total
        assert real_memory.semantic.get(mid).reinforcement_count == total
        reloaded = episodic_module.EpisodicMemoryStore().get_by_id(mid)
        assert reloaded.review_count == total

    def test_parallel_feedback_matches_sequential_ema(self, real_memory):
        mid = real_memory.store.store(
            "rajan", "feedback target",
            hash_embed("feedback target"))["memory_id"]

        def fn():
            for _ in range(OPS):
                real_memory.store.record_feedback(
                    agent_id="rajan", memory_id=mid, positive=True)

        errors = run_concurrently([fn] * THREADS)

        assert errors == []
        n = THREADS * OPS
        expected = 1 - 0.5 * 0.7 ** n
        assert real_memory.episodic.get_by_id(mid).agent_feedback == \
            pytest.approx(expected)


# ─── Scheduler maintenance during live traffic ─────────────

class TestSchedulerUnderLoad:

    def test_cycles_run_during_store_retrieve_and_ageing(self, real_memory):
        """
        The real scheduler loop (tiny interval) runs decay + cleanup while
        writers store, readers retrieve, and an "ager" pushes some fresh
        memories past forgetting (the /simulate_time mutation). Every
        aged memory must end up gone from ALL tiers, no un-aged memory may
        be lost, and the scheduler must not record a single failure.
        """
        mo = real_memory.orchestrator
        sched = MemoryScheduler(real_memory.store, interval_seconds=0.005)
        aged_ids, keep_ids = [], []
        lock = threading.Lock()

        def writer(t):
            def fn():
                for n in range(OPS):
                    c = content_for("rajan", t, n)
                    mid = real_memory.store.store(
                        "rajan", c, hash_embed(c))["memory_id"]
                    with lock:
                        keep_ids.append(mid)
            return fn

        def ager():
            for n in range(OPS):
                c = content_for("ghost", 0, n)
                mid = real_memory.store.store(
                    "ghost", c, hash_embed(c))["memory_id"]
                with mo.episodic_lock:
                    ep = mo.episodic_memory.get_by_id(mid)
                    ep.last_reviewed_at -= timedelta(days=90)
                    mo.episodic_memory._save_episode(ep)
                with lock:
                    aged_ids.append(mid)

        def reader():
            for _ in range(OPS):
                real_memory.store.retrieve(
                    agent_id="rajan", query="note",
                    embedding=hash_embed("note"))
                sched.metrics()
                sched.is_running()

        sched.start()
        try:
            errors = run_concurrently(
                [writer(t) for t in range(4)] + [ager] + [reader] * 2)
            # let at least one more full cycle see the last aged items
            start = sched.total_cycles
            deadline = time.time() + 10
            while sched.total_cycles < start + 2 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            sched.stop()

        assert errors == []
        assert not sched.is_running()
        assert sched.total_failures == 0
        assert sched.total_cycles >= 3

        ram = assert_tiers_consistent(real_memory)
        assert set(keep_ids) <= set(ram), "live memory lost by cleanup"
        for mid in aged_ids:
            assert mid not in ram
            assert real_memory.semantic.get(mid) is None, "orphan in Qdrant"


# ─── Full pipeline in parallel ─────────────────────────────

class TestParallelConversations:

    def test_parallel_routes_stay_isolated(self, real_orchestrator):
        orch = real_orchestrator()
        orch.client.reply = lambda req: "ack " + req["messages"][-1]["content"]
        agents = ["rajan", "alice", "bob", "carol"]
        markers = {a: f"marker{a}" for a in agents}

        def talk(agent):
            def fn():
                for n in range(6):
                    orch.route(f"{agent} says {markers[agent]} item {n}",
                               agent_id=agent)
            return fn

        errors = run_concurrently([talk(a) for a in agents])

        assert errors == []
        assert len(orch.client.requests) == len(agents) * 6
        for req in orch.client.requests:
            speaker = req["messages"][-1]["content"].split(" ", 1)[0]
            text = req["system"] + " ".join(
                m["content"] for m in req["messages"])
            for other in agents:
                if other != speaker:
                    assert markers[other] not in text, (speaker, other)
        for agent in agents:
            wm = orch.memory_store.get_context(agent, limit=50)
            assert len(wm) == 12
            assert all(markers[agent] in m["content"] for m in wm)
