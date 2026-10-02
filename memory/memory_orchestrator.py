"""
FINAL HARDENED
ENTERPRISE-GRADE
MULTI-AGENT MEMORY ORCHESTRATOR
"""

import json
import logging
import math
import threading
import uuid

from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 384

from memory.working_memory import VALID_ROLES  # noqa: E402
from memory.retrieval_config import (  # noqa: E402
    excluded_roles,
    min_relevance_score
)


def _is_forgotten(ep) -> bool:
    check = getattr(ep, "is_forgotten", None)
    if not callable(check):
        return False
    try:
        return bool(check()) is True
    except Exception:
        return False


class MemoryOrchestrator:
    """
    Coordinates:
    - working memory
    - episodic memory
    - semantic memory
    - reinforcement
    - cleanup
    - retrieval fusion
    - token budgeting

    Design Goals
    ────────────
    - thread safety
    - scoped memory isolation
    - rollback handling
    - bounded retrieval
    - scalable concurrency
    - production-safe validation
    """

    def __init__(
        self,
        working_memory,
        episodic_memory,
        semantic_memory
    ):

        self.working_memory = (
            working_memory
        )

        self.episodic_memory = (
            episodic_memory
        )

        self.semantic_memory = (
            semantic_memory
        )

        # tier-level locks

        self.working_lock = (
            threading.RLock()
        )

        self.episodic_lock = (
            threading.RLock()
        )

        self.semantic_lock = (
            threading.RLock()
        )

    # ─────────────────────────────────────────────────────
    # Store
    # ─────────────────────────────────────────────────────

    def store(
        self,
        agent_id: str,
        content: str,
        embedding: List[float],
        context: Optional[
            Dict[str, Any]
        ] = None
    ) -> Dict[str, str]:

        if context is None:
            context = {}

        # Validate EVERYTHING before the first write. Working memory is
        # written first and never rolled back, so any input rejected by a
        # later tier (Qdrant rejects NaN/bool vectors; SQLite's int(shared)
        # rejects "true"/"no") used to leave a half-stored memory behind.
        self._validate_embedding(
            embedding
        )
        self._validate_context(
            context
        )

        role = context.get(
            "role",
            "user"
        )

        memory_id = str(
            uuid.uuid4()
        )

        episodic_id = None

        try:

            # ─────────────────────────────────────────
            # Working Memory
            # ─────────────────────────────────────────

            # NOTE:
            # Working memory intentionally
            # does not participate in rollback.
            #
            # It is:
            # - transient
            # - bounded
            # - non-persistent
            # - eventually overwritten

            with self.working_lock:

                self.working_memory.add(
                    agent_id=agent_id,
                    role=role,
                    content=content
                )

            # ─────────────────────────────────────────
            # Episodic Memory
            # ─────────────────────────────────────────

            with self.episodic_lock:

                episodic_id = (
                    self.episodic_memory.add(
                        memory_id=memory_id,
                        agent_id=agent_id,
                        content=content,
                        context={
                            **context,

                            "source_agent": (
                                agent_id
                            ),

                            "confidence": (
                                context.get(
                                    "confidence",
                                    1.0
                                )
                            ),

                            "shared": (
                                context.get(
                                    "shared",
                                    False
                                )
                            )
                        },
                        embedding=embedding
                    )
                )

            # ─────────────────────────────────────────
            # Semantic Memory
            # ─────────────────────────────────────────

            with self.semantic_lock:

                semantic_id = (
                    self.semantic_memory.add(
                        memory_id=memory_id,
                        content=content,
                        embedding=embedding,
                        metadata={

                            "agent_id": (
                                agent_id
                            ),

                            "episodic_id": (
                                episodic_id
                            ),

                            "shared": (
                                context.get(
                                    "shared",
                                    False
                                )
                            ),

                            "confidence": (
                                context.get(
                                    "confidence",
                                    1.0
                                )
                            ),

                            "role": role
                        }
                    )
                )

            logger.info(
                "stored memory "
                "memory_id=%s "
                "agent_id=%s",
                memory_id,
                agent_id
            )

            return {
                "memory_id": memory_id,
                "episodic_id": episodic_id,
                "semantic_id": semantic_id
            }

        except Exception:

            logger.exception(
                "store failed "
                "memory_id=%s",
                memory_id
            )

            # NOTE:
            # Per-tier locking allows
            # higher concurrency than
            # a global orchestrator lock.
            #
            # This creates a small TOCTOU
            # window where another thread
            # may briefly observe an
            # episodic entry before
            # rollback removes it.
            #
            # This is considered acceptable
            # because:
            # - rollback events are rare
            # - eventual consistency is
            #   sufficient for cognition
            # - semantic tier acts as the
            #   authoritative retrieval layer

            if episodic_id is not None:

                try:

                    with self.episodic_lock:

                        self.episodic_memory.delete(
                            episodic_id
                        )

                except Exception:

                    logger.exception(
                        "rollback failed "
                        "episodic_id=%s",
                        episodic_id
                    )

            raise

    # ─────────────────────────────────────────────────────
    # Retrieve
    # ─────────────────────────────────────────────────────

    def retrieve(
        self,
        agent_id: str,
        query: str,
        embedding: List[float],
        top_k: int = 5,
        token_budget: int = 4000,
        min_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval: semantic (Qdrant ANN) + episodic (cached
        vectors, retention-aware), then fusion.

        Only memories whose cosine similarity to the query is at least
        `min_score` (default RETRIEVAL_MIN_SCORE, 0.30) are returned;
        everything below is treated as irrelevant — not injected into
        the prompt and therefore not reinforced. The assistant's own
        replies are excluded unless RETRIEVE_ASSISTANT_REPLIES=true.
        """

        self._validate_embedding(
            embedding
        )

        if min_score is None:
            min_score = min_relevance_score()
        skip_roles = excluded_roles()

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        # ── Semantic Retrieval ──
        with self.semantic_lock:
            semantic_results = (
                self.semantic_memory.search_multi(
                    embedding=embedding,
                    limit=top_k,
                    agent_id=agent_id,
                    min_similarity=min_score,
                    exclude_roles=skip_roles
                )
            )

        # ── Episodic Retrieval ──
        with self.episodic_lock:
            episodic_scored = (
                self.episodic_memory
                .grounded_retrieve_scored(
                    query=query,
                    embedding=embedding,
                    top_k=top_k,
                    agent_id=agent_id,
                    min_score=min_score,
                    exclude_roles=skip_roles
                )
            )

        # A memory that has decayed past the forgetting threshold but
        # hasn't been pruned yet must not come back through the vector
        # index either (the episodic tier already skips it).
        if semantic_results:
            with self.episodic_lock:
                live = []
                for mem in semantic_results:
                    ep = self.episodic_memory.get_by_id(mem.memory_id)
                    if ep is not None and _is_forgotten(ep):
                        continue
                    live.append(mem)
            semantic_results = live

        episodic_results = [ep for _, ep in episodic_scored]

        # ── Working Context ──
        with self.working_lock:
            working_context = (
                self.working_memory
                .as_messages(
                    agent_id=agent_id,
                    limit=5
                )
            )

        # ── Relevance scores (cross-tier, by shared memory id) ──
        scores: Dict[str, float] = {}
        for score, ep in episodic_scored:
            scores[ep.episode_id] = max(
                score, scores.get(ep.episode_id, 0.0)
            )
        for mem in semantic_results:
            sim = getattr(mem, "similarity", None)
            if sim is None:
                sim = getattr(mem, "score", None) or 0.0
            scores[mem.memory_id] = max(
                float(sim), scores.get(mem.memory_id, 0.0)
            )

        # ── Fusion Layer ──
        fused_results = (
            self._budgeted_fusion(
                episodic_results,
                semantic_results,
                token_budget,
                scores=scores
            )
        )

        return {
            "query": query,
            "working_memory": (
                working_context
            ),
            "episodic_memory": (
                episodic_results
            ),
            "semantic_memory": (
                semantic_results
            ),
            "fused_memory": (
                fused_results
            ),
            "scores": {
                self._memory_key(m): round(
                    scores.get(self._memory_key(m), 0.0), 4
                )
                for m in fused_results
            },
            "min_score": min_score
        }

    # ─────────────────────────────────────────────────────
    # Context
    # ─────────────────────────────────────────────────────

    def get_context(
        self,
        agent_id: str,
        limit: int = 10
    ):

        with self.working_lock:

            return (
                self.working_memory
                .as_messages(
                    agent_id=agent_id,
                    limit=limit
                )
            )

    # ─────────────────────────────────────────────────────
    # Reinforcement
    # ─────────────────────────────────────────────────────

    def reinforce(
        self,
        agent_id: str,
        memory_id: str
    ):

        with self.episodic_lock:

            self.episodic_memory.recall(
                episode_id=memory_id
            )

        with self.semantic_lock:

            self.semantic_memory.reinforce(
                memory_id
            )

        logger.info(
            "reinforced memory "
            "agent_id=%s "
            "memory_id=%s",
            agent_id,
            memory_id
        )

    # ─────────────────────────────────────────────────────
    # Feedback
    # ─────────────────────────────────────────────────────

    def record_feedback(
        self,
        memory_id: str,
        positive: bool
    ) -> bool:
        """
        Record agent/human feedback on a memory (episodic tier
        only — this feeds Episode.agent_feedback in the DMS
        priority_score() formula). Returns True if a matching
        episode was found and updated.
        """

        with self.episodic_lock:

            ep = self.episodic_memory.record_feedback(
                episode_id=memory_id,
                positive=positive
            )

        logger.info(
            "recorded feedback "
            "memory_id=%s positive=%s found=%s",
            memory_id,
            positive,
            ep is not None
        )

        return ep is not None

    def record_task_outcome(
        self,
        memory_id: str,
        success: bool
    ) -> bool:
        """
        Record whether a memory contributed to a successful task
        outcome (feeds Episode.task_success_rate). Returns True if
        a matching episode was found and updated.
        """

        with self.episodic_lock:

            ep = self.episodic_memory.record_task_outcome(
                episode_id=memory_id,
                success=success
            )

        logger.info(
            "recorded task outcome "
            "memory_id=%s success=%s found=%s",
            memory_id,
            success,
            ep is not None
        )

        return ep is not None

    # ─────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────

    def cleanup(self) -> int:
        """
        Cleanup forgotten memories
        across all tiers.

        EXPECTS:
            prune_forgotten()
            -> List[str]
        """

        with self.episodic_lock:

            removed_ids = (
                self.episodic_memory
                .prune_forgotten()
            )

        if not isinstance(
            removed_ids,
            list
        ):

            raise TypeError(
                "prune_forgotten() "
                "must return List[str]"
            )

        # NOTE:
        # Episodic and semantic cleanup
        # are intentionally non-atomic.
        #
        # A concurrent store using a
        # recycled memory_id between
        # these lock boundaries could
        # theoretically create an
        # orphaned semantic entry.
        #
        # This is considered acceptable
        # because:
        # - memory_id uses UUID4
        # - collision probability is
        #   astronomically negligible

        self._delete_semantic(removed_ids)

        logger.info(
            "cleanup complete "
            "removed=%s",
            len(removed_ids)
        )

        return len(removed_ids)

    def _delete_semantic(
        self,
        memory_ids: List[str]
    ):
        """Best-effort: one failed delete never blocks the rest."""

        with self.semantic_lock:

            for memory_id in memory_ids:

                try:

                    self.semantic_memory.delete(
                        memory_id
                    )

                except Exception:

                    logger.exception(
                        "semantic cleanup failed "
                        "memory_id=%s",
                        memory_id
                    )

    # ─────────────────────────────────────────────────────
    # Decay
    # ─────────────────────────────────────────────────────

    def apply_decay(self):

        with self.episodic_lock:

            removed_ids = self.episodic_memory.apply_decay()

        # Episodic apply_decay() removes forgotten episodes itself, so a
        # following cleanup() finds nothing to prune and never reaches
        # its semantic-delete loop. Delete the vectors here too, or
        # forgotten memories resurface via semantic search.
        if isinstance(removed_ids, list):
            self._delete_semantic(removed_ids)

    # ─────────────────────────────────────────────────────
    # Fusion Layer
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _memory_key(memory):
        # Episode objects carry `episode_id`; SemanticMemory objects
        # carry `memory_id`. store() writes both tiers under the same
        # uuid, so either attribute is the cross-tier identity.
        return getattr(
            memory, "memory_id", None
        ) or getattr(
            memory, "episode_id", None
        )

    def _budgeted_fusion(
        self,
        episodic_results,
        semantic_results,
        token_budget: int,
        scores: Optional[Dict[str, float]] = None
    ):
        """
        Deduplicated, relevance-ranked memory fusion with token
        budgeting.

        1. Dedup across tiers by shared memory id (content as a
           fallback). When a memory is found in both tiers the
           Episode object is kept (it carries retention/feedback).
        2. Rank by relevance (cosine similarity, when `scores` is
           given), breaking ties by retention so a fresher memory
           wins over a fading one with the same relevance. Without
           scores, tier order is kept (episodic, then semantic).
        3. Fill the token budget best-first. A single memory that is
           too large to fit is skipped rather than ending the fill —
           otherwise one huge memory ranked first would leave the
           prompt with no memory context at all.
        """

        combined = []

        seen = set()

        for memory in (
            episodic_results
            + semantic_results
        ):

            memory_id = self._memory_key(memory)

            content = getattr(
                memory,
                "content",
                ""
            )

            dedup_key = (
                memory_id
                or str(content).strip()
            )

            if dedup_key in seen:
                continue

            seen.add(dedup_key)

            combined.append(memory)

        if scores:

            def rank(item):
                index, memory = item
                relevance = scores.get(
                    self._memory_key(memory), 0.0
                )
                retention_fn = getattr(memory, "retention", None)
                try:
                    retention = (
                        float(retention_fn())
                        if callable(retention_fn) else 0.0
                    )
                except Exception:
                    retention = 0.0
                return (-round(relevance, 6), -retention, index)

            combined = [
                memory for _, memory
                in sorted(enumerate(combined), key=rank)
            ]

        current_tokens = 0

        filtered = []

        for memory in combined:

            content = getattr(
                memory,
                "content",
                ""
            )

            estimated_tokens = (
                self._estimate_tokens(
                    content
                )
            )

            if (
                current_tokens
                + estimated_tokens
                > token_budget
            ):
                # Too big for what's left — try the next (smaller)
                # memory instead of discarding everything after it.
                continue

            filtered.append(memory)

            current_tokens += (
                estimated_tokens
            )

        return filtered

    # ─────────────────────────────────────────────────────
    # Token Estimation
    # ─────────────────────────────────────────────────────

    def _estimate_tokens(
        self,
        text: str
    ) -> int:
        """
        Rough token estimation.

        Future upgrade path:
        - tiktoken
        - Anthropic tokenizer
        """

        text = text.strip()

        if not text:
            return 0

        words = len(
            text.split()
        )

        return int(words * 1.3)

    # ─────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────

    def _validate_embedding(
        self,
        embedding: List[float]
    ):

        if not isinstance(
            embedding,
            list
        ):

            raise ValueError(
                "embedding must be a list"
            )

        if not embedding:

            raise ValueError(
                "embedding cannot be empty"
            )

        if len(embedding) != (
            EXPECTED_EMBEDDING_DIM
        ):

            raise ValueError(
                f"expected embedding "
                f"dimension "
                f"{EXPECTED_EMBEDDING_DIM}, "
                f"got {len(embedding)}"
            )

        if not all(
            isinstance(
                value,
                (int, float)
            )
            and not isinstance(value, bool)
            for value in embedding
        ):

            raise ValueError(
                "embedding must contain "
                "numeric values"
            )

        if not all(
            math.isfinite(value)
            for value in embedding
        ):

            raise ValueError(
                "embedding must not contain "
                "NaN or infinite values"
            )

    def _validate_context(
        self,
        context: Dict[str, Any]
    ):

        if not isinstance(context, dict):
            raise ValueError(
                "context must be a dict"
            )

        role = context.get("role", "user")
        if role not in VALID_ROLES:
            raise ValueError(
                f"context role must be one of "
                f"{sorted(VALID_ROLES)}"
            )

        if not isinstance(
            context.get("shared", False),
            bool
        ):
            raise ValueError(
                "context 'shared' must be a boolean"
            )

        # EpisodicMemoryStore persists context as JSON; a value json
        # can't encode (datetime, set, custom object) used to be saved as
        # str(context) — fine until restart, then ep.context reloads as a
        # *string* and metrics() crashes for every agent.
        try:
            json.dumps(context)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"context must be JSON-serializable: {e}"
            )

        confidence = context.get("confidence", 1.0)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
        ):
            raise ValueError(
                "context 'confidence' must be a finite number"
            )
    def metrics(self, agent_id: str = None) -> Dict[str, Any]:
        """
        Returns live memory panel data.
        """

        with self.episodic_lock:
            all_episodes = self.episodic_memory.get_all()

        # filter by agent if specified
        episodes = [
            ep for ep in all_episodes
            if (
                agent_id is None
                or ep.agent_id == agent_id
                or ep.shared
            )
            and ep.context.get("role") != "assistant"
        ]

        total = len(episodes)
        if total == 0:
            return {
                "total_memories": 0,
                "avg_retention": 0.0,
                "avg_stability": 0.0,
                "avg_priority": 0.0,
                "forgotten_count": 0,
                "memories": []
            }

        retentions = []
        stabilities = []
        priorities = []
        forgotten = 0
        memory_list = []

        for ep in episodes:
            r = ep.retention()        # live R(t)
            p = ep.priority_score()   # live DMS score
            s = ep.stability_hours

            retentions.append(r)
            stabilities.append(s)
            priorities.append(p)

            if ep.is_forgotten():
                forgotten += 1

            memory_list.append({
                "id": ep.episode_id,
                "content": str(ep.content)[:80],
                "retention": round(r, 3),
                "stability_hours": round(s, 2),
                "priority_score": round(p, 3),
                "access_count": ep.review_count,
                "agent_id": ep.agent_id,
                "shared": ep.shared,
                "agent_feedback": round(ep.agent_feedback, 3),
                "task_success_rate": round(ep.task_success_rate, 3),
            })

        return {
            "total_memories": total,
            "avg_retention": round(
                sum(retentions) / total, 3
            ),
            "avg_stability": round(
                sum(stabilities) / total, 2
            ),
            "avg_priority": round(
                sum(priorities) / total, 3
            ),
            "forgotten_count": forgotten,
            "memories": memory_list
        }
