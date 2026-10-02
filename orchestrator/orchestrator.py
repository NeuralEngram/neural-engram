# orchestrator.py

"""
Orchestrator

Coordinates task routing across agents.
Now routes through PlannerAgent
before ExecutorAgent.

Flow
────
User query
    ↓
Orchestrator.route()
    ↓
PlannerAgent.plan()
    ↓  (intent detection + optional memory retrieval)
ExecutorAgent.execute()
    ↓
Response
"""

import logging

from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

import anthropic
from dotenv import load_dotenv
import os
load_dotenv()

from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemoryStore
from memory.semantic_memory import SemanticMemoryStore
from memory.memory_orchestrator import MemoryOrchestrator
from memory.memory_store import MemoryStore
from memory.scheduler import MemoryScheduler
from memory.notifier import CaregiverNotifier, detect_alerts
from accounts.store import AccountStore

from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent     # ← Anika's agent


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Startup validation
# ─────────────────────────────────────────────────────────────

# Anything left over from .env.example, or any other value that's
# obviously not a real key, should stop startup rather than let
# anthropic.Anthropic() construct successfully and only fail (or
# silently misbehave) on the first real API call.
_ANTHROPIC_KEY_PLACEHOLDERS = {
    "", "your-anthropic-api-key", "your_anthropic_api_key",
    "changeme", "change-me", "sk-ant-xxxx",
}


# Default chat model. Override with CLAUDE_MODEL in .env so a model
# rename/retirement never needs a code change.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"


def _resolve_claude_model() -> str:
    return os.getenv("CLAUDE_MODEL", "").strip() or DEFAULT_CLAUDE_MODEL


def _validate_anthropic_api_key() -> None:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if (
        not key
        or key.lower() in _ANTHROPIC_KEY_PLACEHOLDERS
        or key.lower().startswith("your-")
    ):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set (or is still the "
            ".env.example placeholder). Set a real Anthropic API "
            "key in your environment or .env file before starting "
            "the orchestrator."
        )


# Anchor point for resolving on-disk storage paths — see
# _resolve_qdrant_storage_path() below for why this matters.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default local persistence location if QDRANT_STORAGE_PATH isn't set at
# all. Previously an unset QDRANT_STORAGE_PATH silently meant "in-memory,
# lost on every restart" — fine as a documented gap before this hardening
# pass, but the whole point of this pass is that local runs persist
# without anyone having to remember to configure that first. Persistence
# is now the default; in-memory mode is still available, but has to be
# requested explicitly (QDRANT_STORAGE_PATH= empty string).
_DEFAULT_QDRANT_STORAGE_PATH = os.path.join(_PROJECT_ROOT, "qdrant_data")


def _resolve_qdrant_storage_path() -> Optional[str]:
    """
    Resolve QDRANT_STORAGE_PATH the same way regardless of the
    directory the process happens to be launched from.

    - Unset entirely            -> default persistent path under the
                                    project root (qdrant_data/).
    - Set to an empty string    -> explicit opt-out; in-memory mode
                                    (SemanticMemoryStore logs its own
                                    warning about data loss on restart).
    - Set to a relative path    -> resolved against the project root,
                                    NOT the process's current working
                                    directory, so "./qdrant_data" means
                                    the same directory no matter where
                                    `python`/`uvicorn` was invoked from.
    - Set to an absolute path   -> used as-is.
    """

    raw = os.getenv("QDRANT_STORAGE_PATH")

    if raw is None:
        return _DEFAULT_QDRANT_STORAGE_PATH

    if raw.strip() == "":
        return None

    return raw if os.path.isabs(raw) else os.path.join(_PROJECT_ROOT, raw)


# ─────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinates agents and memory stack.

    Agent pipeline
    ──────────────
    PlannerAgent  ← routes + pre-retrieves
    ExecutorAgent ← executes + stores

    Note: agents/memory_agent.py::MemoryAgent is a standalone,
    independently-tested lifecycle-layer class (see
    tests/test_memory_agent.py) but is not part of this live
    pipeline — nothing here calls it, so it's intentionally not
    instantiated below. Wire it in (and route requests through
    it) if/when something actually needs its behavior; until
    then MemoryStore is the real access layer every endpoint and
    agent uses directly.
    """

    def __init__(self):

        # ── Fail fast if the Anthropic key is missing/placeholder ──
        _validate_anthropic_api_key()

        # ── Embedding model ──────────────────
        self.embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # ── Anthropic client ─────────────────
        self.client = anthropic.Anthropic()
        self.model = _resolve_claude_model()

        # ── Memory stack ─────────────────────
        working_memory  = WorkingMemory(capacity=20)
        episodic_memory = EpisodicMemoryStore()

        qdrant_storage_path = _resolve_qdrant_storage_path()
        logger.info(
            "[Orchestrator] Qdrant storage_path resolved to %s",
            qdrant_storage_path if qdrant_storage_path else "<in-memory>"
        )
        semantic_memory = SemanticMemoryStore(
            storage_path=qdrant_storage_path
        )

        memory_orchestrator = MemoryOrchestrator(
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory
        )

        self.memory_store = MemoryStore(
            orchestrator=memory_orchestrator
        )

        # ── Caregiver notifications ──────────
        # Reuses the same critical-keyword/low-retention
        # rule the /alerts endpoint already exposed —
        # this just adds an actual notification on top.
        self.notifier = CaregiverNotifier()

        # ── Application accounts ─────────────
        # Users, caregivers, reminders, alert history. Each user's decay
        # speed (Alzheimer's severity) is applied to new memories.
        try:
            self.accounts = AccountStore()
            self.apply_account_decay_multipliers()
        except Exception:
            logger.exception("[Orchestrator] account store unavailable")
            self.accounts = None
        self._notify_agent_ids = [
            a.strip()
            for a in os.getenv(
                "NOTIFY_AGENT_IDS", "rajan"
            ).split(",")
            if a.strip()
        ]

        # ── Scheduler ────────────────────────
        self.scheduler = MemoryScheduler(
            memory_store=self.memory_store,
            interval_seconds=300,
            on_cycle_complete=self.check_caregiver_alerts
        )
        self.scheduler.start()

        # ── Agents ───────────────────────────

        # ExecutorAgent first (PlannerAgent wraps it)
        self.executor = ExecutorAgent(
            memory_store=self.memory_store,
            embed_fn=self._embed,
            llm_fn=self._llm
        )

        # PlannerAgent wraps ExecutorAgent
        self.planner = PlannerAgent(
            memory_store=self.memory_store,
            executor=self.executor,
            embed_fn=self._embed
        )

        logger.info(
            "[Orchestrator] Initialized. "
            "Pipeline: PlannerAgent → ExecutorAgent"
        )

    # ─────────────────────────────────────────────────────────
    # Embed
    # ─────────────────────────────────────────────────────────

    def _embed(
        self,
        text: str
    ) -> List[float]:
        """
        Generate embedding for text.
        """

        return (
            self.embed_model
            .encode(text)
            .tolist()
        )

    # ─────────────────────────────────────────────────────────
    # LLM
    # ─────────────────────────────────────────────────────────

    def _llm(
        self,
        task: str,
        context: List[Dict[str, str]]
    ) -> str:
        """
        Generate response via Claude.
        """

        # Filter out system messages from context
        # to pass them properly to the API
        system_parts = [
            m["content"]
            for m in context
            if m["role"] == "system"
        ]

        non_system = [
            m for m in context
            if m["role"] != "system"
        ]

        system_prompt = (
            "You are a helpful assistant "
            "with long-term memory. "
            "Use the provided context "
            "to answer accurately.\n\n"
            + "\n\n".join(system_parts)
        ).strip()

        messages = non_system + [
            {
                "role": "user",
                "content": task
            }
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=messages
        )

        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text

        return ""

    # ─────────────────────────────────────────────────────────
    # Route — now goes through PlannerAgent
    # ─────────────────────────────────────────────────────────
    def route(
        self,
        task: str,
        agent_id: str = "default"
    ) -> Dict[str, Any]:

        logger.info(
            "[Orchestrator] Routing task: %r agent_id=%s",
            task[:60],
            agent_id
        )

        return self.planner.plan(
            task=task,
            agent_id=agent_id
        )


    # ─────────────────────────────────────────────────────────
    # Caregiver Alert Check
    # ─────────────────────────────────────────────────────────

    def apply_account_decay_multipliers(self):
        """Push every account's decay multiplier into the episodic tier."""
        if getattr(self, "accounts", None) is None:
            return
        episodic = self.memory_store.orchestrator.episodic_memory
        for username, multiplier in self.accounts.decay_multipliers().items():
            episodic.set_agent_decay_multiplier(username, multiplier)

    def _monitored_agent_ids(self) -> List[str]:
        ids = list(self._notify_agent_ids)
        if getattr(self, "accounts", None) is not None:
            for username in self.accounts.alzheimers_usernames():
                if username not in ids:
                    ids.append(username)
        return ids

    def check_caregiver_alerts(self, event=None):
        """
        Called by MemoryScheduler after every maintenance cycle.

        - Checks each monitored patient (NOTIFY_AGENT_IDS plus every
          Alzheimer's account) for critical/warning memories and
          notifies caregivers — the patient's own caregiver when they
          have turned on email alerts, otherwise ALERT_EMAIL_TO.
        - Records alert status/history and a daily retention snapshot
          per account for the caregiver dashboard.

        Never raises — a notification failure should not affect the
        memory maintenance cycle itself.
        """

        try:
            with self.memory_store.orchestrator.episodic_lock:
                episodes = (
                    self.memory_store.orchestrator
                    .episodic_memory.get_all()
                )

            accounts = getattr(self, "accounts", None)
            alzheimers = (
                set(accounts.alzheimers_usernames()) if accounts else set()
            )

            for agent_id in self._monitored_agent_ids():
                alerts = detect_alerts(episodes, agent_id)
                if agent_id in alzheimers:
                    try:
                        accounts.sync_alerts(agent_id, alerts)
                    except Exception:
                        logger.exception("alert history update failed")
                if alerts:
                    recipients = (
                        accounts.alert_recipients(agent_id)
                        if agent_id in alzheimers else []
                    )
                    self.notifier.notify(
                        agent_id, alerts, to_addrs=recipients or None
                    )

            if accounts is not None:
                self._record_snapshots(episodes)

        except Exception:
            logger.exception(
                "[Orchestrator] Caregiver alert check failed"
            )

    def _record_snapshots(self, episodes):
        """Today's average retention per account (for trend charts)."""
        by_user: Dict[str, List[Any]] = {}
        for ep in episodes:
            ctx = ep.context if isinstance(ep.context, dict) else {}
            if ctx.get("role") == "assistant":
                continue
            by_user.setdefault(ep.agent_id, []).append(ep)
        for username in self.accounts.member_usernames():
            eps = by_user.get(username, [])
            if not eps:
                continue
            retentions = [ep.retention() for ep in eps]
            self.accounts.record_snapshot(
                username,
                sum(retentions) / len(retentions),
                len(eps),
                sum(1 for r in retentions if r < 0.5),
            )

    # Backward-compatible private alias (older callers/tests).
    _check_caregiver_alerts = check_caregiver_alerts

    # ─────────────────────────────────────────────────────────
    # Stop
    # ─────────────────────────────────────────────────────────

    def stop(self):
        """
        Shutdown orchestrator cleanly.
        """

        self.scheduler.stop()

        # Close the vector store explicitly so an on-disk Qdrant releases
        # its lock cleanly instead of in __del__ at interpreter teardown
        # (which logged "ImportError: sys.meta_path is None" on every
        # shutdown).
        self.memory_store.orchestrator.semantic_memory.close()

        logger.info(
            "[Orchestrator] Stopped."
        )