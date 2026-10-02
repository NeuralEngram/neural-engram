# memory_agent.py

"""
Production-Grade Memory Agent

Responsibilities
────────────────
- memory lifecycle coordination
- retrieval orchestration
- reinforcement routing
- decay orchestration
- cleanup orchestration
- memory access abstraction

Architecture
────────────
Planner/Executor Agents
            ↓
        MemoryAgent
            ↓
       MemoryStore
            ↓
    MemoryOrchestrator
            ↓
working / episodic / semantic
"""

import logging

from typing import Any, Dict, List, Optional

from agents.base_agent import (
    BaseAgent
)

from memory.memory_store import (
    MemoryStore
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Memory Agent
# ─────────────────────────────────────────────────────────────

class MemoryAgent(BaseAgent):
    """
    Concrete cognitive memory controller.

    Coordinates:
    - memory storage
    - retrieval
    - reinforcement
    - decay
    - cleanup

    MemoryAgent exists as the
    specialized lifecycle layer
    above BaseAgent.

    IMPORTANT
    ─────────
    By default, MemoryAgent uses
    a shared global agent_id:

        "memory_agent"

    This is intentional for
    centralized cognitive memory.

    If multi-tenant isolation or
    per-session cognition is needed,
    provide unique agent_ids.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        agent_id: str = (
            "memory_agent"
        )
    ):

        super().__init__(
            agent_id=agent_id,
            memory_store=memory_store
        )

        logger.info(
            "MemoryAgent initialized "
            "agent_id=%s",
            self.agent_id
        )

    # ─────────────────────────────────────────────────────────
    # Store Memory
    # ─────────────────────────────────────────────────────────

    def store_memory(
        self,
        content: str,
        embedding: List[float],
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Store memory across all tiers.

        Returns:
            mapping of memory IDs
        """

        memory_ids = (
            super().store_memory(
                content=content,
                embedding=embedding,
                context=context
            )
        )

        logger.info(
            "stored memory "
            "agent_id=%s "
            "memory_id=%s",
            self.agent_id,
            memory_ids.get(
                "memory_id"
            )
        )

        return memory_ids

    # ─────────────────────────────────────────────────────────
    # Retrieve Memories
    # ─────────────────────────────────────────────────────────

    def retrieve_memories(
        self,
        query: str,
        embedding: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories.

        Retrieval alone does not
        automatically reinforce memory.

        Reinforcement must occur only
        after successful usage.
        """

        results = (
            super().retrieve_memories(
                query=query,
                embedding=embedding,
                top_k=top_k
            )
        )

        logger.info(
            "retrieved memories "
            "agent_id=%s "
            "query=%s "
            "top_k=%s",
            self.agent_id,
            query,
            top_k
        )

        return results

    # ─────────────────────────────────────────────────────────
    # Reinforcement
    # ─────────────────────────────────────────────────────────

    def reinforce_memory(
        self,
        memory_id: str
    ) -> None:
        """
        Reinforce successfully used memory.
        """

        logger.info(
            "reinforcing memory "
            "agent_id=%s "
            "memory_id=%s",
            self.agent_id,
            memory_id
        )

        super().reinforce_memory(
            memory_id=memory_id
        )

    # ─────────────────────────────────────────────────────────
    # Working Context
    # ─────────────────────────────────────────────────────────

    def get_context(
        self,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Retrieve formatted working memory
        context window.
        """

        context = (
            super().get_context(
                limit=limit
            )
        )

        logger.info(
            "retrieved context "
            "agent_id=%s "
            "messages=%s",
            self.agent_id,
            len(context)
        )

        return context

    # ─────────────────────────────────────────────────────────
    # Decay
    # ─────────────────────────────────────────────────────────

    def apply_decay(
        self
    ) -> None:
        """
        Apply Ebbinghaus forgetting decay
        across episodic memories.
        """

        logger.info(
            "applying decay "
            "agent_id=%s",
            self.agent_id
        )

        super().apply_decay()

    # ─────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────

    def cleanup_memories(
        self
    ) -> int:
        """
        Remove forgotten memories.

        Returns:
            number of removed memories
        """

        removed = (
            super()
            .cleanup_memories()
        )

        logger.info(
            "cleanup complete "
            "agent_id=%s "
            "removed=%s",
            self.agent_id,
            removed
        )

        return removed

    # ─────────────────────────────────────────────────────────
    # High-Level Cognitive Helpers
    # ─────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        embedding: List[float],
        role: str = "user",

        # IMPORTANT:
        # None default is intentional.
        #
        # Avoids mutable default
        # argument bugs from {}.

        extra_context: Optional[
            Dict[str, Any]
        ] = None
    ) -> Dict[str, str]:
        """
        Convenience helper for memory storage.
        """

        context = {
            "role": role
        }

        if extra_context:

            context.update(
                extra_context
            )

        return self.store_memory(
            content=content,
            embedding=embedding,
            context=context
        )

    # ─────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        embedding: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Lightweight retrieval helper.

        Unlike think(), recall()
        performs no additional
        cognitive introspection
        or fused-memory logging.

        Intended for:
        - simple retrieval
        - direct lookup
        - low-overhead access
        """

        return self.retrieve_memories(
            query=query,
            embedding=embedding,
            top_k=top_k
        )

    # ─────────────────────────────────────────────────────────

    def think(
        self,
        query: str,
        embedding: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Cognitive retrieval operation.

        Retrieves:
        - working context
        - episodic memories
        - semantic memories
        - fused memory ranking

        think() differs from recall()
        by explicitly modeling a
        higher-level cognition phase
        with fused-memory inspection
        and additional observability.
        """

        results = self.retrieve_memories(
            query=query,
            embedding=embedding,
            top_k=top_k
        )

        fused = results.get(
            "fused_memory",
            []
        )

        logger.info(
            "cognitive retrieval "
            "agent_id=%s "
            "fused_results=%s",
            self.agent_id,
            len(fused)
        )

        return results