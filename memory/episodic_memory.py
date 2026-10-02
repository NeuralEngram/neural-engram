# episodic_memory.py

"""
FINAL HARDENED
ENTERPRISE-GRADE
MULTI-AGENT EPISODIC MEMORY STORE

Features
────────
- SQLite persistence
- Ebbinghaus forgetting
- adaptive reinforcement
- shared/private memories
- cross-agent consensus
- replay prioritisation
- production-safe caching
- thread-safe connection pooling
- semantic retrieval support
- importance scoring
- multi-agent cognition
"""

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import uuid
import time

import numpy as np

from collections import OrderedDict

from dataclasses import (
    dataclass,
    field
)

from datetime import (
    datetime,
    timedelta,
    timezone
)

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional
)

import anthropic

from sentence_transformers import (
    SentenceTransformer
)

from memory.retrieval_config import (
    excluded_roles,
    min_relevance_score
)

from memory.ebbinghaus import (
    DEFAULT_STABILITY_HOURS,
    compute_retention,
    is_memory_forgotten,
    reinforce_memory,
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Small, fast model used only to score how important a new memory is.
# Override with CLAUDE_IMPORTANCE_MODEL in .env.
IMPORTANCE_MODEL = (
    os.getenv("CLAUDE_IMPORTANCE_MODEL", "").strip()
    or "claude-haiku-4-5-20251001"
)


# ─────────────────────────────────────────────────────────────
# Lazy Dependency Loading
# ─────────────────────────────────────────────────────────────

_client = None

_model = None

_client_lock = threading.Lock()

_model_lock = threading.Lock()


def get_client():

    global _client

    if _client is None:

        with _client_lock:

            if _client is None:

                _client = anthropic.Anthropic()

    return _client


def get_embedding_model():

    global _model

    if _model is None:

        with _model_lock:

            if _model is None:

                _model = SentenceTransformer(
                    "all-MiniLM-L6-v2"
                )

    return _model


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.20

MIN_RESULTS_REQUIRED = 1

DEFAULT_LIMIT = 100

REPLAY_HORIZON_HOURS = 6

# DB_PATH was previously a bare relative filename ("multi_agent_memory.db"),
# which SQLite resolves against the process's current working directory at
# launch — not against this file's location. Two runs started from
# different directories (a different terminal, an IDE run config, a
# systemd unit with no WorkingDirectory set, etc.) would silently open two
# different files, and the "restart" would appear to lose everything even
# though nothing was actually deleted. Anchor it to the project root
# instead, with an optional MAEM_DB_PATH override for anyone who wants a
# different location (e.g. a shared data volume).
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

_raw_db_path = os.getenv("MAEM_DB_PATH")
if _raw_db_path:
    DB_PATH = (
        _raw_db_path
        if os.path.isabs(_raw_db_path)
        else os.path.join(_PROJECT_ROOT, _raw_db_path)
    )
else:
    DB_PATH = os.path.join(_PROJECT_ROOT, "multi_agent_memory.db")

logger.info("EpisodicMemoryStore DB_PATH resolved to %s", DB_PATH)

# Per-agent decay-speed override, e.g. for demoing forgetting
# behaviour faster for one patient/agent without waiting real
# time. Configure via AGENT_DECAY_MULTIPLIERS="agent_id:mult,..."
# (default preserves the previous hardcoded demo behaviour of
# decaying agent_id "rajan" 3x faster than everyone else — this
# was previously hardcoded by name directly in add() below).
# A multiplier > 1.0 divides effective stability, so retention
# drops faster; 1.0 is normal decay speed.


def _load_agent_decay_multipliers() -> Dict[str, float]:
    raw = os.getenv("AGENT_DECAY_MULTIPLIERS", "rajan:3.0")
    result: Dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        agent, _, mult = pair.partition(":")
        agent = agent.strip()
        try:
            value = float(mult.strip())
        except ValueError:
            logger.warning(
                "AGENT_DECAY_MULTIPLIERS: could not parse "
                "multiplier for %r — skipping", agent
            )
            continue
        # Episode.retention() divides stability by this, so 0 raises
        # ZeroDivisionError and a negative value raises ValueError —
        # and grounded_retrieve() evaluates retention for EVERY agent's
        # episodes, so one bad entry would break retrieval system-wide.
        if not (value > 0 and math.isfinite(value)):
            logger.warning(
                "AGENT_DECAY_MULTIPLIERS: multiplier for %r must be "
                "finite and > 0 (got %r) — skipping", agent, value
            )
            continue
        result[agent] = value
    return result


# shared memory weights

W_ACCESS_FREQ = 0.15

W_TASK_IMPORTANCE = 0.18

W_AGENT_FEEDBACK = 0.32

W_TASK_SUCCESS = 0.27

W_CONSENSUS = 0.08


assert 0.0 < W_CONSENSUS < 1.0

assert abs(
    W_ACCESS_FREQ +
    W_TASK_IMPORTANCE +
    W_AGENT_FEEDBACK +
    W_TASK_SUCCESS +
    W_CONSENSUS -
    1.0
) < 1e-9, (
    "shared weights must sum to 1.0"
)


# private memory weights

_PRIVATE_TOTAL = (
    1.0 - W_CONSENSUS
)

W_ACCESS_FREQ_P = (
    W_ACCESS_FREQ / _PRIVATE_TOTAL
)

W_TASK_IMPORTANCE_P = (
    W_TASK_IMPORTANCE / _PRIVATE_TOTAL
)

W_AGENT_FEEDBACK_P = (
    W_AGENT_FEEDBACK / _PRIVATE_TOTAL
)

W_TASK_SUCCESS_P = (
    W_TASK_SUCCESS / _PRIVATE_TOTAL
)


# ─────────────────────────────────────────────────────────────
# Thread-Local SQLite Connections
# ─────────────────────────────────────────────────────────────

_thread_local = threading.local()


def get_connection(
    db_path: str = DB_PATH
) -> sqlite3.Connection:
    """
    Per-thread SQLite connection.

    Connections are isolated:
    - per thread
    - per db_path
    """

    attr = (
        f"connection_{db_path}"
        .replace("/", "_")
        .replace(".", "_")
    )

    conn = getattr(
        _thread_local,
        attr,
        None
    )

    if conn is None:

        conn = sqlite3.connect(
            db_path,
            check_same_thread=False
        )

                # Two threads opening a brand-new DB at the same instant can
        # race on the WAL switch; SQLite returns "database is locked"
        # immediately here instead of waiting, so retry briefly.
        for attempt in range(20):
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                break
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 19:
                    raise
                time.sleep(0.05)

        conn.execute(
            "PRAGMA synchronous=NORMAL;"
        )

        setattr(
            _thread_local,
            attr,
            conn
        )

    return conn


# ─────────────────────────────────────────────────────────────
# Importance Cache
# ─────────────────────────────────────────────────────────────

_importance_cache = OrderedDict()

_importance_cache_lock = (
    threading.Lock()
)

MAX_IMPORTANCE_CACHE = 5000


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class RecallStatus(Enum):

    UPDATED = "updated"

    FORGOTTEN = "forgotten"

    NOT_FOUND = "not_found"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def safe_json_dumps(
    obj: Any
) -> str:

    try:

        return json.dumps(obj)

    except (
        TypeError,
        ValueError
    ):

        return json.dumps(
            str(obj)
        )


def safe_json_loads(
    s: Any
) -> Any:

    try:

        return json.loads(s)

    except (
        TypeError,
        ValueError,
        json.JSONDecodeError
    ):

        return s


def simple_embedding(
    text: str
) -> List[float]:

    model = get_embedding_model()

    return model.encode(
        text
    ).tolist()


def cosine_similarity(
    a: List[float],
    b: List[float]
) -> float:

    if not a or not b:
        return 0.0

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    na = math.sqrt(
        sum(x * x for x in a)
    )

    nb = math.sqrt(
        sum(x * x for x in b)
    )

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


# ─────────────────────────────────────────────────────────────
# Importance Scoring
# ─────────────────────────────────────────────────────────────

def compute_importance(
    text: str,
    agent_id: str
) -> float:

    cache_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]

    cache_key = (
        f"{agent_id}:{cache_hash}"
    )

    with _importance_cache_lock:

        cached = (
            _importance_cache.get(
                cache_key
            )
        )

        if cached is not None:

            # maintain LRU ordering

            _importance_cache.move_to_end(
                cache_key
            )

            return cached

    # ─────────────────────────────────────────
    # Fast Heuristic Paths
    # ─────────────────────────────────────────
    #
    # Heuristic scores are intentionally
    # cached to avoid repeated evaluation
    # for ultra-common short messages.
    #

    if len(text.split()) <= 3:

        score = 0.2

    elif text.strip().endswith("?"):

        score = 0.15

    else:

        # ─────────────────────────────────────
        # LLM Importance Scoring
        # ─────────────────────────────────────

        try:

            client = get_client()

            resp = client.messages.create(
                model=IMPORTANCE_MODEL,

                max_tokens=5,

                timeout=5.0,

                system=(
                    f"You are agent "
                    f"'{agent_id}'. "
                    "Rate importance "
                    "from 0.0 to 1.0. "
                    "Only return a number."
                ),

                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )

            score = float(
                resp.content[0]
                .text
                .strip()
            )

            score = max(
                0.0,
                min(score, 1.0)
            )

        except Exception as exc:

            logger.warning(
                "compute_importance failed "
                "for agent '%s': %s",
                agent_id,
                exc
            )

            score = _heuristic_importance(
                text
            )

    # ─────────────────────────────────────────
    # Cache Writeback
    # ─────────────────────────────────────────

    with _importance_cache_lock:

        # bounded cache growth
        #
        # eviction happens BEFORE insert
        # to maintain strict upper bounds

        if len(_importance_cache) >= (
            MAX_IMPORTANCE_CACHE
        ):

            logger.warning(
                "importance cache full "
                "evicting oldest entries "
                "cache_size=%d",
                len(_importance_cache)
            )

            eviction_count = int(
                MAX_IMPORTANCE_CACHE * 0.10
            )

            for _ in range(eviction_count):

                _importance_cache.popitem(
                    last=False
                )

        _importance_cache[
            cache_key
        ] = score

        # maintain LRU ordering

        _importance_cache.move_to_end(
            cache_key
        )

    return score


def _heuristic_importance(
    text: str
) -> float:

    text_lower = text.lower()

    high = [
        "i am",
        "i'm",
        "i love",
        "i hate",
        "my goal",
        "i need"
    ]

    low = [
        "okay",
        "thanks",
        "hello",
        "hi"
    ]

    if any(
        s in text_lower
        for s in high
    ):
        return 0.75

    if any(
        s in text_lower
        for s in low
    ):
        return 0.15

    return 0.45


def should_store_memory(
    text: str,
    importance: float
) -> bool:
    """
    Storage thresholds:

    - all memories require:
      importance >= 0.15

    - ultra-short memories
      require:
      importance >= 0.20
    """

    if importance < 0.15:
        return False

    if (
        len(text.split()) <= 2
        and importance < 0.20
    ):
        return False

    return True


# ─────────────────────────────────────────────────────────────
# Episode
# ─────────────────────────────────────────────────────────────

@dataclass
class Episode:

    content: Any

    context: Dict[str, Any]

    agent_id: str

    tags: List[str] = field(
        default_factory=list
    )

    shared: bool = False

    stability_hours: float = (
        DEFAULT_STABILITY_HOURS
    )

    last_reviewed_at: datetime = field(
        default_factory=lambda:
        datetime.now(timezone.utc)
    )

    review_count: int = 0
    decay_multiplier: float = 1.0

    cross_agent_recall_count: int = 0

    episode_id: str = field(
        default_factory=lambda:
        str(uuid.uuid4())
    )

    created_at: datetime = field(
        default_factory=lambda:
        datetime.now(timezone.utc)
    )

    importance: float = 0.6

    agent_feedback: float = 0.5

    task_success_rate: float = 0.5

    def retention(
        self,
        now: Optional[datetime] = None
    ) -> float:

        return compute_retention(
            self.last_reviewed_at,
            self.stability_hours/self.decay_multiplier,

            now
        )

    def is_forgotten(
        self,
        now: Optional[datetime] = None
    ) -> bool:

        return is_memory_forgotten(
            self.retention(now)
        )

    def access_frequency_score(
        self
    ) -> float:

        return min(
            math.log1p(
                self.review_count
            ) / math.log1p(100),
            1.0
        )

    def consensus_score(
        self
    ) -> float:

        if not self.shared:
            return 0.0

        return min(
            math.log1p(
                self.cross_agent_recall_count
            ) / math.log1p(20),
            1.0
        )

    def priority_score(
        self,
        now: Optional[datetime] = None
    ) -> float:
        """
        Replay priority score.

        Retention acts as a global
        decay multiplier.
        """

        af = self.access_frequency_score()

        ti = max(
            0.0,
            min(self.importance, 1.0)
        )

        fb = max(
            0.0,
            min(self.agent_feedback, 1.0)
        )

        ts = max(
            0.0,
            min(self.task_success_rate, 1.0)
        )

        if self.shared:

            cs = self.consensus_score()

            weighted = (
                W_ACCESS_FREQ * af +
                W_TASK_IMPORTANCE * ti +
                W_AGENT_FEEDBACK * fb +
                W_TASK_SUCCESS * ts +
                W_CONSENSUS * cs
            )

        else:

            weighted = (
                W_ACCESS_FREQ_P * af +
                W_TASK_IMPORTANCE_P * ti +
                W_AGENT_FEEDBACK_P * fb +
                W_TASK_SUCCESS_P * ts
            )

        return weighted * self.retention(now)
    

# ─────────────────────────────────────────────────────────────
# EpisodicMemoryStore
# ─────────────────────────────────────────────────────────────

class EpisodicMemoryStore:

    def __init__(self):
        self._episodes: List[Episode] = []
        # episode_id -> unit-normalised embedding. Filled when a memory
        # is stored (the caller already computed the vector) or lazily,
        # once, for episodes loaded from SQLite after a restart — so a
        # query never re-embeds the whole memory store.
        self._vectors: Dict[str, np.ndarray] = {}
        self._agent_decay_multipliers = (
            _load_agent_decay_multipliers()
        )
        self._init_db()
        self._load_from_db()

    def _init_db(self):
        conn = get_connection(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                agent_id TEXT,
                content TEXT,
                context TEXT,
                shared INTEGER,
                stability_hours REAL,
                last_reviewed_at TEXT,
                review_count INTEGER,
                importance REAL,
                agent_feedback REAL,
                task_success_rate REAL,
                decay_multiplier REAL,
                created_at TEXT
            )
        """)
        conn.commit()

    def _load_from_db(self):
        conn = get_connection(DB_PATH)
        cursor = conn.execute("SELECT * FROM episodes")
        rows = cursor.fetchall()
        for row in rows:
            ep = Episode(
                episode_id=row[0],
                agent_id=row[1],
                content=row[2],
                context=self._context_from_db(row[3]),
                shared=bool(row[4]),
                stability_hours=row[5],
                last_reviewed_at=datetime.fromisoformat(row[6]),
                review_count=row[7],
                importance=row[8],
                agent_feedback=row[9],
                task_success_rate=row[10],
                decay_multiplier=row[11],
                created_at=datetime.fromisoformat(row[12])
            )
            self._episodes.append(ep)

    @staticmethod
    def _context_from_db(raw) -> Dict[str, Any]:
        """
        Older rows may hold a non-dict context (safe_json_dumps fell back
        to json.dumps(str(ctx)) for unserialisable values). Keep the raw
        value rather than letting one bad row crash metrics() — which
        calls ep.context.get() — for every agent.
        """
        value = safe_json_loads(raw)
        if isinstance(value, dict):
            return value
        logger.warning(
            "episode context in DB is not a dict — preserving as "
            "{'_unparsed': ...}"
        )
        return {"_unparsed": value}

    def _save_episode(self, ep: Episode):
        conn = get_connection(DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO episodes VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            ep.episode_id,
            ep.agent_id,
            str(ep.content),
            safe_json_dumps(ep.context),
            int(ep.shared),
            ep.stability_hours,
            ep.last_reviewed_at.isoformat(),
            ep.review_count,
            ep.importance,
            ep.agent_feedback,
            ep.task_success_rate,
            ep.decay_multiplier,
            ep.created_at.isoformat()
        ))
        conn.commit()

    def set_agent_decay_multiplier(self, agent_id: str, multiplier: float):
        """Decay speed for this agent's NEW memories (>1 = forgets faster)."""
        if multiplier and multiplier > 0:
            self._agent_decay_multipliers[agent_id] = float(multiplier)

    def save(self, ep: Episode):
        """Public: persist an updated episode to SQLite."""
        self._save_episode(ep)

    def simulate_time_passage(
        self,
        agent_id: str,
        hours: float
    ) -> int:
        """
        Demo helper: pretend `hours` have passed for one agent by
        moving each of its episodes' last-review time back, then
        persisting it. Callers must hold the episodic lock.
        Returns how many episodes were aged.
        """
        aged = 0
        for ep in self.get_all():
            if ep.agent_id != agent_id:
                continue
            ep.last_reviewed_at = (
                ep.last_reviewed_at - timedelta(hours=hours)
            )
            self._save_episode(ep)
            aged += 1
            logger.debug("simulate_time: aged %s", str(ep.content)[:30])
        return aged

    def _delete_episode(self, episode_id: str):
        conn = get_connection(DB_PATH)
        conn.execute(
            "DELETE FROM episodes WHERE episode_id = ?",
            (episode_id,)
        )
        conn.commit()
    def add(
        self,
        memory_id: str,
        agent_id: str,
        content: Any,
        context: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        decay_multiplier = self._agent_decay_multipliers.get(
            agent_id, 1.0
        )

        # importance affects stability: S = 24h × (1 + 2·importance).
        # _heuristic_importance() returns 0.15 / 0.45 / 0.75, i.e.
        # S = 31.2h / 45.6h / 60h (evaluation/RESULTS.md, section 1).
        importance = _heuristic_importance(str(content))
        stability = DEFAULT_STABILITY_HOURS * (1 + importance * 2)

        ep = Episode(
            episode_id=memory_id,
            agent_id=agent_id,
            content=content,
            context=context or {},
            shared=(context or {}).get("shared", False),
            decay_multiplier=decay_multiplier,
            stability_hours=stability,
            importance=importance
        )
        # Persist FIRST, then publish to the in-memory list. The old
        # order (append, then save) left an orphan whenever the save
        # raised: the caller (MemoryOrchestrator.store) got an exception
        # and so skipped rollback, but the episode stayed retrievable in
        # RAM — and silently vanished on the next restart.
        self._save_episode(ep)
        self._episodes.append(ep)
        if embedding is not None:
            self._cache_vector(ep.episode_id, embedding)
        return ep.episode_id

    def get_all(self) -> List[Episode]:
        return list(self._episodes)

    def get_by_id(
        self,
        episode_id: str
    ) -> Optional[Episode]:
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def remove(
        self,
        episode_id: str
    ) -> bool:
        before = len(self._episodes)
        self._episodes = [
            ep for ep in self._episodes
            if ep.episode_id != episode_id
        ]
        self._vectors.pop(episode_id, None)
        return len(self._episodes) < before

    def delete(
        self,
        episode_id: str
    ) -> bool:
        self._delete_episode(episode_id)
        return self.remove(episode_id)

    def record_feedback(
        self,
        episode_id: str,
        positive: bool,
        alpha: float = 0.3
    ) -> Optional[Episode]:
        """
        Update Episode.agent_feedback via an exponential moving
        average, nudging it toward 1.0 on positive feedback and
        toward 0.0 on negative feedback.

        alpha controls how much a single feedback event moves the
        running average (0 < alpha <= 1). Persisted immediately.
        """
        ep = self.get_by_id(episode_id)
        if ep is None:
            return None

        target = 1.0 if positive else 0.0
        ep.agent_feedback = ep.agent_feedback + alpha * (
            target - ep.agent_feedback
        )
        ep.agent_feedback = max(0.0, min(ep.agent_feedback, 1.0))

        self._save_episode(ep)
        return ep

    def record_task_outcome(
        self,
        episode_id: str,
        success: bool,
        alpha: float = 0.3
    ) -> Optional[Episode]:
        """
        Update Episode.task_success_rate via an exponential moving
        average, nudging it toward 1.0 on success and toward 0.0
        on failure.

        alpha controls how much a single outcome moves the running
        average (0 < alpha <= 1). Persisted immediately.
        """
        ep = self.get_by_id(episode_id)
        if ep is None:
            return None

        target = 1.0 if success else 0.0
        ep.task_success_rate = ep.task_success_rate + alpha * (
            target - ep.task_success_rate
        )
        ep.task_success_rate = max(0.0, min(ep.task_success_rate, 1.0))

        self._save_episode(ep)
        return ep

    def recall(
        self,
        episode_id: str
    ) -> Optional[Episode]:
        ep = self.get_by_id(episode_id)
        if ep:
            ep.review_count += 1
            ep.last_reviewed_at = datetime.now(timezone.utc)
            ep.stability_hours = reinforce_memory(
                stability_hours=ep.stability_hours,
                quality=1.0,
                review_count=ep.review_count
            )
            self._save_episode(ep)
        return ep

    # ─────────────────────────────────────────────────────
    # Vector cache
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _normalise(vector) -> Optional[np.ndarray]:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return None
        norm = float(np.linalg.norm(arr))
        if norm == 0.0 or not np.isfinite(norm):
            return None
        return arr / norm

    def _cache_vector(self, episode_id: str, embedding) -> None:
        vec = self._normalise(embedding)
        if vec is not None:
            self._vectors[episode_id] = vec

    def _vector_for(self, ep: "Episode") -> Optional[np.ndarray]:
        vec = self._vectors.get(ep.episode_id)
        if vec is None:
            # Episode loaded from SQLite (or stored without a vector):
            # embed it once and keep it.
            try:
                vec = self._normalise(simple_embedding(str(ep.content)))
            except Exception:
                logger.exception(
                    "embedding failed for episode %s", ep.episode_id
                )
                vec = None
            if vec is not None:
                self._vectors[ep.episode_id] = vec
        return vec

    # ─────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────

    def grounded_retrieve_scored(
        self,
        query: str,
        embedding: List[float],
        top_k: int = 5,
        agent_id: Optional[str] = None,
        min_score: Optional[float] = None,
        exclude_roles: Optional[List[str]] = None
    ) -> List[tuple]:
        """
        Relevance-ranked episodic retrieval.

        Returns [(cosine_similarity, Episode), ...], best first.
        Only live (not forgotten) memories owned by `agent_id` or
        shared, at or above `min_score`, and not written by a role in
        `exclude_roles`. Similarity is one matrix-vector product over
        cached unit vectors — no re-embedding per query.
        """
        if top_k <= 0:
            return []
        query_vec = self._normalise(embedding) if embedding else None
        if query_vec is None:
            return []
        if min_score is None:
            min_score = min_relevance_score()
        if exclude_roles is None:
            exclude_roles = excluded_roles()
        skip_roles = set(exclude_roles)

        candidates = []
        vectors = []
        for ep in self._episodes:
            if agent_id is not None and ep.agent_id != agent_id and not ep.shared:
                continue
            ctx = ep.context if isinstance(ep.context, dict) else {}
            if ctx.get("role") in skip_roles:
                continue
            if ep.is_forgotten():
                continue
            vec = self._vector_for(ep)
            if vec is None or vec.shape != query_vec.shape:
                continue
            candidates.append(ep)
            vectors.append(vec)

        if not candidates:
            return []

        scores = np.vstack(vectors) @ query_vec
        keep = np.nonzero(scores >= min_score)[0]
        if keep.size == 0:
            return []
        # best first; stable for equal scores
        order = keep[np.argsort(-scores[keep], kind="stable")][:top_k]
        return [(float(scores[i]), candidates[i]) for i in order]

    def grounded_retrieve(
        self,
        query: str,
        embedding: List[float],
        top_k: int = 5,
        agent_id: Optional[str] = None,
        min_score: Optional[float] = None,
        exclude_roles: Optional[List[str]] = None
    ) -> List[Episode]:
        return [
            ep for _, ep in self.grounded_retrieve_scored(
                query=query,
                embedding=embedding,
                top_k=top_k,
                agent_id=agent_id,
                min_score=min_score,
                exclude_roles=exclude_roles
            )
        ]

    def apply_decay(self) -> List[str]:
        """
        Drop forgotten episodes from RAM and SQLite. Returns the removed
        ids so the caller can remove the matching semantic vectors —
        without that, "forgotten" memories stayed in Qdrant and kept
        coming back through semantic retrieval into the LLM prompt.
        """
        forgotten = [
            ep.episode_id
            for ep in self._episodes
            if ep.is_forgotten()
        ]
        self._episodes = [
            ep for ep in self._episodes
            if not ep.is_forgotten()
        ]
        for episode_id in forgotten:
            self._delete_episode(episode_id)
            self._vectors.pop(episode_id, None)
        return forgotten

    def prune_forgotten(self) -> List[str]:
        forgotten = [
            ep.episode_id
            for ep in self._episodes
            if ep.is_forgotten()
        ]
        self._episodes = [
            ep for ep in self._episodes
            if not ep.is_forgotten()
        ]
        for episode_id in forgotten:
            self._delete_episode(episode_id)
            self._vectors.pop(episode_id, None)
        return forgotten