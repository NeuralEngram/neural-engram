"""
Improved Base Agent
"""

import logging

from typing import Any, Dict, List, Optional

from memory.memory_store import (
    MemoryStore
)

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Shared base for all agents.

    Responsibilities
    ────────────────
    - memory storage
    - memory retrieval
    - reinforcement
    - working context access
    - maintenance delegation
    - observability access

    Architecture
    ────────────
    BaseAgent acts as the unified
    cognitive interface between:

        agents
            ↓
        MemoryStore
            ↓
        orchestrator
            ↓
    memory subsystems
    """

    def __init__(
        self,
        agent_id: str,
        memory_store: MemoryStore
    ):

        agent_id = agent_id.strip()

        if not agent_id:

            raise ValueError(
                "agent_id cannot be empty"
            )

        self.agent_id = agent_id

        self.memory_store = memory_store

        logger.info(
            "BaseAgent initialized "
            "agent_id=%s",
            self.agent_id
        )

    # ─────────────────────────────────────────────────────
    # Store Memory
    # ─────────────────────────────────────────────────────

    def store_memory(
        self,
        content: str,
        embedding: List[float],
        agent_id: Optional[str] = None,
        context: Optional[
            Dict[str, Any]
        ] = None
    ) -> Dict[str, str]:
        """
        Store memory safely.

        Memory is automatically scoped
        to this agent instance.
        """

        return self.memory_store.store(
            agent_id=agent_id or self.agent_id,
            content=content,
            embedding=embedding,
            context=context
        )

    # ─────────────────────────────────────────────────────
    # Retrieve Memories
    # ─────────────────────────────────────────────────────

    def retrieve_memories(
        self,
        query: str,
        embedding: List[float],
        agent_id: Optional[str] = None,
        top_k: int = 5,
        token_budget: int = 4000
    ) -> Dict[str, Any]:
        """
        Retrieve memories relevant
        to this agent.

        Retrieval includes:
        - working memory
        - episodic memory
        - semantic memory
        - fused memory ranking
        """

        return self.memory_store.retrieve(
            agent_id=agent_id or self.agent_id,
            query=query,
            embedding=embedding,
            top_k=top_k,
            token_budget=token_budget
        )

    # ─────────────────────────────────────────────────────
    # Reinforce Memory
    # ─────────────────────────────────────────────────────

    def reinforce_memory(
        self,
        memory_id: str,
        agent_id: Optional[str] = None
    ) -> None:
        """
        Reinforce successfully-used memory.

        Reinforcement is scoped to this
        agent instance by default. Pass
        agent_id explicitly when reinforcing
        on behalf of a different agent/session
        (e.g. PlannerAgent reinforcing memories
        for the conversational agent it is
        routing on behalf of), otherwise the
        reinforcement is logged/attributed to
        this agent's own id instead of the
        conversation's.
        """

        self.memory_store.reinforce(
            agent_id=agent_id or self.agent_id,
            memory_id=memory_id
        )

    # ─────────────────────────────────────────────────────
    # Working Context
    # ─────────────────────────────────────────────────────

    def get_context(
        self,
        limit: int = 10,
        agent_id: Optional[str] = None
    ):
        """
        Retrieve working memory context.

        Defaults to this agent's own scope,
        but callers acting on behalf of a
        different agent/session (e.g. an
        ExecutorAgent executing a task for
        agent_id="rajan") must pass agent_id
        explicitly, or they will silently
        read this agent's own (usually empty)
        working memory instead of the actual
        conversation's.
        """

        return self.memory_store.get_context(
            agent_id=agent_id or self.agent_id,
            limit=limit
        )

    # ─────────────────────────────────────────────────────
    # Global Decay
    # ─────────────────────────────────────────────────────

    def apply_decay(
        self
    ):
        """
        Apply Ebbinghaus forgetting
        decay across ALL agents.

        IMPORTANT
        ─────────
        Decay is a global maintenance
        operation and is NOT scoped
        to self.agent_id.
        """

        self.memory_store.apply_decay()

    # ─────────────────────────────────────────────────────
    # Global Cleanup
    # ─────────────────────────────────────────────────────

    def cleanup_memories(
        self
    ) -> int:
        """
        Remove forgotten memories
        across ALL agents.

        IMPORTANT
        ─────────
        Cleanup is a global maintenance
        operation and is NOT scoped
        to self.agent_id.
        """

        return self.memory_store.cleanup()

    # ─────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────

    def memory_metrics(
        self
    ):
        """
        Retrieve memory system metrics.

        Returns observability data for:
        - working memory
        - episodic memory
        - semantic memory
        - scheduler state
        """

        return (
            self.memory_store
            .metrics()
        )