# executor_agent.py

"""
Executor Agent

Executes tasks using memory context.
"""

import logging

from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent

from memory.memory_store import (
    MemoryStore
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Executor Agent
# ─────────────────────────────────────────────────────────────

class ExecutorAgent(BaseAgent):
    """
    Executes tasks using retrieved
    memory context and generates
    responses via LLM.

    Responsibilities
    ────────────────
    - task execution
    - memory retrieval
    - context augmentation
    - response generation
    - interaction persistence
    - cognitive orchestration
    """

    def __init__(
        self,
        memory_store: MemoryStore,

        embed_fn: Callable[
            [str],
            List[float]
        ],

        llm_fn: Callable[
            [
                str,
                List[Dict[str, str]]
            ],
            str
        ],

        agent_id: str = (
            "executor_agent"
        )
    ):

        super().__init__(
            agent_id=agent_id,
            memory_store=memory_store
        )

        self.embed_fn = embed_fn

        self.llm_fn = llm_fn

        logger.info(
            "ExecutorAgent initialized "
            "agent_id=%s",
            self.agent_id
        )

    # ─────────────────────────────────────────────────────────
    # Memory Context Builder
    # ─────────────────────────────────────────────────────────

    def _build_memory_context(
        self,
        memories: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Convert retrieved memories into
        LLM-compatible context.

        IMPORTANT
        ─────────
        Uses fused_memory instead of
        raw semantic/episodic results.

        This preserves:
        - orchestrator ranking
        - deduplication
        - token budgeting
        - relevance ordering
        """

        memory_lines = []

        fused_memories = memories.get(
            "fused_memory",
            []
        )

        for memory in fused_memories:

            memory_lines.append(
                f"Memory: "
                f"{memory.content}"
            )

        if not memory_lines:

            return []

        return [{
            "role": "system",
            "content": (
                "Relevant memory context:\n\n"
                + "\n".join(memory_lines)
            )
        }]

    # ─────────────────────────────────────────────────────────
    # Execute
    # ─────────────────────────────────────────────────────────

    def execute(
        self,
        task: str,
        agent_id: str = "default",

        context: Optional[
            Dict[str, Any]
        ] = None,

        retrieved_memories: Optional[
            Dict[str, Any]
        ] = None,

        embedding: Optional[
            List[float]
        ] = None
    ) -> Dict[str, Any]:
        """
        Execute task lifecycle.

        retrieved_memories / embedding:
            When PlannerAgent has already embedded the task and
            retrieved memories, it passes both in so this turn does
            exactly one embedding of the task and one retrieval.
            Called directly (no planner), the executor does both
            itself.

        Steps
        ─────
        1. Embed task
        2. Retrieve memories
        3. Retrieve working context
        4. Build augmented context
        5. Generate response
        6. Store interaction
        7. Return execution result

        context:
            Metadata attached to the
            stored user memory entry.

            Does NOT affect LLM input.
        """
        logger.debug(
            "[ExecutorAgent] execute called agent_id=%s",
            agent_id
        )
        task = task.strip()

        if not task:

            raise ValueError(
                "task cannot be empty"
            )

        if context is None:

            context = {
                "role": "user"
            }

        if embedding is None:
            embedding = self.embed_fn(
                task
            )

        # IMPORTANT:
        # Retrieve BEFORE storing
        # to avoid self-retrieval.

        if retrieved_memories is not None:
            memories = retrieved_memories
        else:
            memories = self.retrieve_memories(
                query=task,
                embedding=embedding,
                agent_id=agent_id
            )

        # IMPORTANT: must pass the conversation's agent_id here,
        # not rely on self.agent_id (this executor's own id) —
        # otherwise this always reads an empty working-memory
        # buffer instead of the real conversation's context.
        working_context = (
            self.get_context(
                agent_id=agent_id
            )
        )

        retrieved_context = (
            self._build_memory_context(
                memories
            )
        )

        # ─────────────────────────────────────
        # Merge System Messages
        # ─────────────────────────────────────

        system_messages = [
            message
            for message
            in working_context
            if message["role"]
            == "system"
        ]

        non_system_messages = [
            message
            for message
            in working_context
            if message["role"]
            != "system"
        ]

        system_parts = [
            message["content"]
            for message
            in system_messages
        ]

        if retrieved_context:

            system_parts.append(
                retrieved_context[0][
                    "content"
                ]
            )

        augmented_context = []

        if system_parts:

            augmented_context.append({
                "role": "system",
                "content": (
                    "\n\n".join(
                        system_parts
                    )
                )
            })

        augmented_context.extend(
            non_system_messages
        )

        # ─────────────────────────────────────
        # LLM Execution
        # ─────────────────────────────────────

        try:

            response = self.llm_fn(
                task,
                augmented_context
            )
        except Exception as error:
            logger.error(
                "[ExecutorAgent] "
                "LLM call failed: %s",
                error,
                exc_info=True
            )
            raise

        # ─────────────────────────────────────
        # Store Interaction
        # ─────────────────────────────────────
        #
        # IMPORTANT:
        # Storage failures should NOT
        # suppress successful LLM output.
        #
        # The user already received a
        # valid cognitive response.
        #
        # Memory persistence is treated
        # as best-effort durability.
        #

        try:

            response_embedding = (
                self.embed_fn(
                    response
                )
            )
            logger.debug(
                "[ExecutorAgent] storing agent_id=%s task=%s",
                agent_id,
                task[:40]
            )
            if not task.strip().endswith("?"):
                self.store_memory(
                content=task,
                embedding=embedding,
                agent_id=agent_id,
                context=context
            )

            self.store_memory(
                content=response,
                embedding=(
                    response_embedding
                ),
                agent_id=agent_id,
                context={
                    "role": "assistant"
                }
            )

        except Exception as error:

            logger.error(
                "[ExecutorAgent] "
                "Memory storage failed: %s",
                error,
                exc_info=True
            )

        # ─────────────────────────────────────
        # Observability
        # ─────────────────────────────────────

        fused_count = len(
            memories.get(
                "fused_memory",
                []
            )
        )

        logger.info(
            "[ExecutorAgent] "
            "Executed task=%r | "
            "fused=%d | "
            "response_len=%d",
            task,
            fused_count,
            len(response)
        )

        return {
            "task": task,
            "response": response,
            "augmented_context": (
                augmented_context
            ),
            "retrieved_memories": (
                memories
            )
        }

    # ─────────────────────────────────────────────────────────
    # Respond
    # ─────────────────────────────────────────────────────────

    def respond(
        self,
        task: str
    ) -> str:
        """
        Convenience wrapper around
        execute().

        Returns only response text.

        IMPORTANT
        ─────────
        respond() does NOT expose
        metadata/context injection.

        Use execute() when:
        - attaching metadata
        - storing custom context
        - requiring full execution
          diagnostics
        """

        result = self.execute(
            task=task
        )

        return result["response"]