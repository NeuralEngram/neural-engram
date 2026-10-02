"""
Unified Multi-Agent Memory Store
"""

from typing import Any, Dict, List, Optional

from memory.memory_orchestrator import (
    MemoryOrchestrator
)


class MemoryStore:
    """
    Unified memory access layer.
    """

    def __init__(
        self,
        orchestrator: MemoryOrchestrator
    ):

        self.orchestrator = (
            orchestrator
        )

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
        """
        Store memory across all tiers.
        """

        if not agent_id.strip():

            raise ValueError(
                "agent_id cannot be empty"
            )

        if not content.strip():

            raise ValueError(
                "content cannot be empty"
            )

        return self.orchestrator.store(
            agent_id=agent_id,
            content=content,
            context=context,
            embedding=embedding
        )

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
        Retrieve relevant memories.

        min_score: relevance cut-off override (default:
        RETRIEVAL_MIN_SCORE from the environment).
        """

        kwargs = {}
        if min_score is not None:
            kwargs["min_score"] = min_score

        return self.orchestrator.retrieve(
            agent_id=agent_id,
            query=query,
            embedding=embedding,
            top_k=top_k,
            token_budget=token_budget,
            **kwargs
        )

    # ─────────────────────────────────────────────────────

    def get_context(
        self,
        agent_id: str,
        limit: int = 10
    ):
        """
        Retrieve scoped working memory.
        """

        return self.orchestrator.get_context(
            agent_id=agent_id,
            limit=limit
        )

    # ─────────────────────────────────────────────────────

    def reinforce(
        self,
        agent_id: str,
        memory_id: str
    ):
        """
        Reinforce memory.
        """

        self.orchestrator.reinforce(
            agent_id=agent_id,
            memory_id=memory_id
        )

    # ─────────────────────────────────────────────────────

    def record_feedback(
        self,
        agent_id: str,
        memory_id: str,
        positive: bool
    ) -> bool:
        """
        Record agent/human feedback on a memory.
        """

        return self.orchestrator.record_feedback(
            memory_id=memory_id,
            positive=positive
        )

    # ─────────────────────────────────────────────────────

    def record_task_outcome(
        self,
        agent_id: str,
        memory_id: str,
        success: bool
    ) -> bool:
        """
        Record whether a memory contributed to a successful
        task outcome.
        """

        return self.orchestrator.record_task_outcome(
            memory_id=memory_id,
            success=success
        )

    # ─────────────────────────────────────────────────────

    def cleanup(self) -> int:

        return (
            self.orchestrator
            .cleanup()
        )

    # ─────────────────────────────────────────────────────

    def apply_decay(self):

        self.orchestrator.apply_decay()

    # ─────────────────────────────────────────────────────

    def metrics(self):

        if hasattr(
            self.orchestrator,
            "metrics"
        ):
            return (
                self.orchestrator
                .metrics()
            )

        return {}