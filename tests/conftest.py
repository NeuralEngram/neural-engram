# tests/conftest.py
"""
Shared fixtures for tests that run against REAL memory objects.

Why this exists
───────────────
Three bugs from the debug pass (working-memory agent scoping, executor
context scoping, planner reinforcement id) and one found while writing
these tests (fusion dedup never matching Episode objects) all survived a
fully passing suite because the tests replaced WorkingMemory /
EpisodicMemoryStore / SemanticMemoryStore / get_context with MagicMocks.
A MagicMock accepts any attribute and any call signature, so it cannot
tell you that a real object lacks `.memory_id` or that content landed in
the wrong agent's buffer.

`real_memory` wires up the genuine stack:

    WorkingMemory → EpisodicMemoryStore (SQLite) → SemanticMemoryStore
    (in-memory Qdrant) → MemoryOrchestrator → MemoryStore

The ONLY things substituted are external resources:
  - the SQLite file (redirected to a per-test tmp_path, never the real
    project-root DB)
  - the sentence-transformer model (replaced by a deterministic
    bag-of-words hash embedding, so tests are fast, offline, and
    similarity is predictable: texts sharing words are similar,
    texts sharing none have cosine 0)
"""

import hashlib
import re
from types import SimpleNamespace

import pytest

import memory.episodic_memory as episodic_module
from memory.memory_orchestrator import (
    EXPECTED_EMBEDDING_DIM,
    MemoryOrchestrator,
)
from memory.memory_store import MemoryStore
from memory.semantic_memory import SemanticMemoryStore
from memory.working_memory import WorkingMemory


# Filler words are dropped so two unrelated sentences don't look
# similar just because both contain "the" or "my".
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "i", "my",
    "me", "you", "your", "to", "of", "and", "or", "in", "on", "at",
    "for", "with", "do", "does", "did", "what", "about", "it", "this",
    "that", "be", "tell", "remember", "recall",
}


def hash_embed(text: str):
    """
    Deterministic 384-dim bag-of-words embedding.

    Stable across processes (md5, not Python's salted hash()).
    """
    vec = [0.0] * EXPECTED_EMBEDDING_DIM
    words = [
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOPWORDS
    ]
    for w in words:
        idx = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[idx % EXPECTED_EMBEDDING_DIM] += 1.0
    if not any(vec):
        # Never hand Qdrant a zero vector (cosine undefined).
        vec[0] = 1.0
    return vec


@pytest.fixture(autouse=True)
def _fast_password_hashing(monkeypatch):
    """Seeding 12 accounts at production PBKDF2 cost is slow; tests use
    the minimum iteration count (the scheme itself is unchanged)."""
    monkeypatch.setenv("MAEM_PASSWORD_ITERATIONS", "1000")


@pytest.fixture(autouse=True)
def _hash_embed_relevance_threshold(monkeypatch):
    """
    hash_embed is bag-of-words: two texts sharing one content word out
    of ~4 score ≈ 0.25, unrelated texts score 0. Real MiniLM embeddings
    put paraphrases at ≈ 0.5–0.8, hence the production default of
    0.30. Calibrate the relevance cut-off to this test embedding's
    scale; tests about the cut-off itself set their own value.
    """
    monkeypatch.setenv("RETRIEVAL_MIN_SCORE", "0.20")


@pytest.fixture
def embed():
    return hash_embed


@pytest.fixture
def real_memory(tmp_path, monkeypatch):
    """
    Fully real memory stack, isolated per test.

    Returns a namespace with: store (MemoryStore), orchestrator
    (MemoryOrchestrator), working, episodic, semantic, db_path.
    """
    db_path = str(tmp_path / "test_memory.db")

    # EpisodicMemoryStore reads the module-level DB_PATH at call time,
    # so patching the module attribute redirects every read/write.
    monkeypatch.setattr(episodic_module, "DB_PATH", db_path)
    monkeypatch.setattr(episodic_module, "simple_embedding", hash_embed)
    # Keep the "rajan:3.0" demo default out of these tests unless a
    # test opts in explicitly.
    monkeypatch.setenv("AGENT_DECAY_MULTIPLIERS", "")

    working = WorkingMemory(capacity=20)
    episodic = episodic_module.EpisodicMemoryStore()
    semantic = SemanticMemoryStore()  # in-memory Qdrant

    orchestrator = MemoryOrchestrator(
        working_memory=working,
        episodic_memory=episodic,
        semantic_memory=semantic,
    )

    return SimpleNamespace(
        store=MemoryStore(orchestrator=orchestrator),
        orchestrator=orchestrator,
        working=working,
        episodic=episodic,
        semantic=semantic,
        db_path=db_path,
    )


# ─── Guard: no test may leave a real class monkeypatched ───
#
# test_memory_agent.py used to do `BaseAgent.store_memory = MagicMock()`
# with no teardown. That leaked into every later test in the process,
# so real-memory agent tests silently stored nothing. This autouse
# guard snapshots the methods of the core classes and fails the
# offending test (and restores the original) if anything is left
# replaced after it finishes. Use monkeypatch / patch.object instead —
# both restore automatically.

from agents.base_agent import BaseAgent  # noqa: E402
from agents.executor_agent import ExecutorAgent  # noqa: E402
from agents.planner_agent import PlannerAgent  # noqa: E402

_GUARDED_CLASSES = (
    BaseAgent, ExecutorAgent, PlannerAgent,
    MemoryStore, MemoryOrchestrator, WorkingMemory,
    episodic_module.EpisodicMemoryStore, SemanticMemoryStore,
)

_ORIGINALS = {
    cls: dict(vars(cls)) for cls in _GUARDED_CLASSES
}


@pytest.fixture(autouse=True)
def _no_leaked_class_patches():
    yield
    leaked = []
    for cls, original in _ORIGINALS.items():
        current = vars(cls)
        for name in set(original) | set(current):
            if original.get(name) is not current.get(name):
                leaked.append(f"{cls.__name__}.{name}")
                if name in original:
                    setattr(cls, name, original[name])
                else:
                    delattr(cls, name)
    if leaked:
        pytest.fail(
            "test left class attributes patched (use monkeypatch or "
            "patch.object so they're restored): " + ", ".join(leaked)
        )


# ─── Real FastAPI app over the real memory stack ───────────
#
# api/main.py builds `orch = Orchestrator()` at import time, which would
# load a sentence-transformer, open the project-root SQLite/Qdrant files
# and start the scheduler. `api_app` swaps ONLY that constructor for a
# thin shell that exposes the real MemoryStore from `real_memory` plus a
# real PlannerAgent → ExecutorAgent pipeline (fake LLM). Every route,
# the auth dependency, CORS middleware and slowapi limiter are the real
# ones from api/main.py.

TEST_API_KEY = "test-maem-key"
ALLOWED_TEST_ORIGIN = "http://localhost:3000"


class _ShellOrchestrator:

    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.executor = ExecutorAgent(
            memory_store=memory_store,
            embed_fn=hash_embed,
            llm_fn=lambda task, ctx: "Okay, noted.",
        )
        self.planner = PlannerAgent(
            memory_store=memory_store,
            executor=self.executor,
            embed_fn=hash_embed,
        )
        self.scheduler = SimpleNamespace(is_running=lambda: False)

    def route(self, task, agent_id="default"):
        return self.planner.plan(task=task, agent_id=agent_id)

    def check_caregiver_alerts(self, event=None):
        pass

    _check_caregiver_alerts = check_caregiver_alerts

    def stop(self):
        pass


@pytest.fixture
def api_app(real_memory, monkeypatch):
    """
    Fresh import of api.main per test. Returns the module; use
    `api_app.app` with fastapi.testclient.TestClient.
    """
    import importlib
    import sys

    import dotenv
    import orchestrator.orchestrator as orch_module

    # A developer's real .env must not leak into these tests.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("MAEM_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("ALLOWED_ORIGINS", ALLOWED_TEST_ORIGIN)
    monkeypatch.setattr(
        orch_module,
        "Orchestrator",
        lambda: _ShellOrchestrator(real_memory.store),
    )

    sys.modules.pop("api.main", None)
    module = importlib.import_module("api.main")
    module.limiter.reset()

    yield module

    module.limiter.reset()
    sys.modules.pop("api.main", None)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


# ═══════════════════════════════════════════════════════════
# Full Orchestrator() for integration tests
# ═══════════════════════════════════════════════════════════
#
# `real_orchestrator(...)` runs the genuine Orchestrator.__init__:
# key validation, memory stack, CaregiverNotifier, MemoryScheduler,
# ExecutorAgent, PlannerAgent — and route() goes through the real
# Orchestrator._llm(), which builds the system prompt and message list
# and calls client.messages.create(). Only external resources change:
#
#   anthropic.Anthropic   → FakeAnthropicClient (records every request)
#   SentenceTransformer   → FakeSentenceTransformer (hash_embed)
#   SQLite path           → tmp_path
#   Qdrant                → in-memory (QDRANT_STORAGE_PATH="")
#   notifications         → email/SMS forced off unless a test enables
#                           them against a fake SMTP server
#
# The background scheduler thread is started by __init__ (real code
# path) and then stopped at once, so its immediate startup cycle can't
# race a test. Tests drive maintenance with orch.scheduler.run_once().

import copy  # noqa: E402

import numpy as np  # noqa: E402


class FakeSentenceTransformer:

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text):
        return np.array(hash_embed(text))


class FakeAnthropicClient:
    """
    Stands in for anthropic.Anthropic(). `requests` holds a deep copy of
    the kwargs of every messages.create() call — i.e. exactly what would
    have been sent to the API.
    """

    def __init__(self, *args, **kwargs):
        self.requests = []
        self.reply = "Okay, noted."
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        text = self.reply(kwargs) if callable(self.reply) else self.reply
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)]
        )

    # Convenience views on the last request
    @property
    def last_system(self):
        return self.requests[-1]["system"]

    @property
    def last_messages(self):
        return self.requests[-1]["messages"]

    def last_request_text(self):
        req = self.requests[-1]
        return req["system"] + "\n" + "\n".join(
            m["content"] for m in req["messages"]
        )


def _isolate_storage(tmp_path, monkeypatch, decay_multipliers=""):
    monkeypatch.setattr(
        episodic_module, "DB_PATH", str(tmp_path / "test_memory.db")
    )
    monkeypatch.setattr(episodic_module, "simple_embedding", hash_embed)
    monkeypatch.setenv("AGENT_DECAY_MULTIPLIERS", decay_multipliers)


@pytest.fixture
def real_orchestrator(tmp_path, monkeypatch):
    """
    Factory: real_orchestrator(decay_multipliers="", notify_agent_ids=
    "rajan", env=None) → a real, fully wired Orchestrator.
    """
    import orchestrator.orchestrator as orch_module

    created = []

    def factory(decay_multipliers="", notify_agent_ids="rajan", env=None):
        _isolate_storage(tmp_path, monkeypatch, decay_multipliers)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
        monkeypatch.setenv("QDRANT_STORAGE_PATH", "")
        monkeypatch.setenv("NOTIFY_AGENT_IDS", notify_agent_ids)
        # A developer .env loaded at import time must never cause real
        # emails/SMS from the test suite.
        monkeypatch.setenv("ENABLE_EMAIL_ALERTS", "false")
        monkeypatch.setenv("ENABLE_SMS_ALERTS", "false")
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)

        monkeypatch.setattr(
            orch_module, "SentenceTransformer", FakeSentenceTransformer
        )
        monkeypatch.setattr(
            orch_module,
            "anthropic",
            SimpleNamespace(Anthropic=FakeAnthropicClient),
        )

        orch = orch_module.Orchestrator()
        orch.scheduler.stop()
        created.append(orch)
        return orch

    yield factory

    for orch in created:
        orch.stop()
