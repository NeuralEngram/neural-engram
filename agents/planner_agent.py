# planner_agent.py

"""
Planner Agent — NeuralEngram

Responsibilities
────────────────
- task analysis (memory-needed or not)
- task decomposition into subtasks
- routing decision (memory → executor)
- memory retrieval orchestration
- response coordination

Architecture
────────────
User Query
    ↓
PlannerAgent.plan()  — detect intent → decide whether / how much to retrieve
    ↓ (skipped for small talk)
BaseAgent.retrieve_memories()   ← the ONLY retrieval in a turn
    ↓ (retrieved memories + query embedding handed over)
ExecutorAgent.execute()
    ↓
Response
"""

import logging
import re

from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.executor_agent import ExecutorAgent

from memory.memory_store import MemoryStore
from memory.retrieval_config import top_k_for_intent


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Intent Detection Signals
# ─────────────────────────────────────────────────────────────

MEMORY_SIGNALS = [
    "remember", "recall", "what did",
    "what do you know", "based on",
    "from before", "earlier", "last time",
    "you said", "we discussed", "history",
    "previous", "forgot", "remind me",
    "what was", "tell me about my",
]

REASONING_SIGNALS = [
    "analyze", "compare", "explain why",
    "evaluate", "reason", "pros and cons",
    "difference between", "how does",
    "why is", "contrast", "assess",
    "break down", "justify",
]

GENERAL_SIGNALS = [
    "what is", "define", "who is",
    "when was", "how do i", "how to",
    "give me", "list", "show me",
    "tell me", "describe",
]


# ─────────────────────────────────────────────────────────────
# Task Intent
# ─────────────────────────────────────────────────────────────

# Questions about the user themself ("what is MY daughter's name",
# "where did I park") need memory even without an explicit
# "remember"-style signal. Word-boundary matched.
PERSONAL_REFERENCE = re.compile(
    r"\b(my|mine|myself|am i|did i|have i|was i|"
    r"where do i|when do i|what do i|who am i)\b"
)

# A message made ONLY of these words is small talk: no retrieval.
SMALL_TALK_WORDS = frozenset({
    "hi", "hello", "hey", "hiya", "yo", "hii", "helo",
    "thanks", "thank", "thx", "ty", "you", "so", "much", "a", "lot",
    "ok", "okay", "k", "kk", "cool", "great", "nice", "awesome",
    "good", "morning", "afternoon", "evening", "night", "day",
    "bye", "goodbye", "see", "later", "cya", "cheers",
    "yes", "yeah", "yep", "no", "nope", "sure", "alright",
    "hmm", "hm", "lol", "haha", "welcome", "fine", "got", "it",
})


class TaskIntent:
    MEMORY     = "memory_retrieval"
    REASONING  = "reasoning"
    GENERAL    = "general_response"
    SMALL_TALK = "small_talk"


def detect_intent(task: str) -> str:
    """
    Score task against signal lists and return the best-fit intent.

    Priority order:
    MEMORY > REASONING > SMALL_TALK > GENERAL

    MEMORY also fires on personal references ("my", "did I",
    "where do I" …). SMALL_TALK only when every word is a
    greeting/acknowledgement ("hi", "ok thanks", "good night").
    """
    lowered = task.lower()

    def score(signals):
        return sum(1 for s in signals if s in lowered)

    if score(MEMORY_SIGNALS) > 0 or PERSONAL_REFERENCE.search(lowered):
        return TaskIntent.MEMORY

    if score(REASONING_SIGNALS) > 0:
        return TaskIntent.REASONING

    words = re.findall(r"[a-z']+", lowered)
    if words and all(w in SMALL_TALK_WORDS for w in words):
        return TaskIntent.SMALL_TALK

    return TaskIntent.GENERAL


# ─────────────────────────────────────────────────────────────
# Planner Agent
# ─────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    Cognitive task planner.

    PlannerAgent sits between the
    Orchestrator and ExecutorAgent.

    It decides:
    - whether memory retrieval is needed
    - what the task intent is
    - how to route the task

    This preserves ExecutorAgent's
    existing interface while adding
    intelligent pre-planning.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        executor: ExecutorAgent,
        embed_fn: Callable[[str], List[float]],
        agent_id: str = "planner_agent"
    ):

        super().__init__(
            agent_id=agent_id,
            memory_store=memory_store
        )

        self.executor  = executor
        self.embed_fn  = embed_fn

        logger.info(
            "PlannerAgent initialized "
            "agent_id=%s",
            self.agent_id
        )

    # ─────────────────────────────────────────────────────────
    # Intent Detection
    # ─────────────────────────────────────────────────────────

    def _detect_intent(
        self,
        task: str
    ) -> str:
        """See detect_intent()."""
        return detect_intent(task)

    # ─────────────────────────────────────────────────────────
    # Memory Needed Check
    # ─────────────────────────────────────────────────────────

    def _requires_memory(
        self,
        intent: str
    ) -> bool:
        """
        Should this turn retrieve long-term memory?

        Everything except pure small talk does — a GENERAL question
        can still be about the user ("where do I live?"), and the
        relevance cut-off keeps unrelated memories out anyway. Small
        talk ("hi", "ok thanks") skips retrieval entirely: no vector
        search, nothing injected, nothing reinforced.
        """
        return intent != TaskIntent.SMALL_TALK

    # ─────────────────────────────────────────────────────────
    # Plan
    # ─────────────────────────────────────────────────────────

    def plan(
        self,
        task: str,
        agent_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full planning pipeline — ONE retrieval per turn.

        1. Detect intent
        2. Decide whether memory is needed and how many memories
           to ask for (top_k by intent)
        3. Embed the task once and retrieve once
        4. Reinforce the memories that passed the relevance cut-off
        5. Hand the retrieved memories AND the query embedding to
           ExecutorAgent, so it neither re-retrieves nor re-embeds
        6. Return the enriched result

        context:
            Metadata for memory storage.
            Passed through to executor.
        """

        task = task.strip()

        if not task:
            raise ValueError(
                "task cannot be empty"
            )

        # ── Phase 1: Intent ──────────────────

        intent       = self._detect_intent(task)
        needs_memory = self._requires_memory(intent)
        top_k        = top_k_for_intent(intent) if needs_memory else 0

        logger.info(
            "[PlannerAgent] "
            "intent=%s | needs_memory=%s | top_k=%d | "
            "task=%r",
            intent,
            needs_memory,
            top_k,
            task[:60]
        )

        # ── Phase 2: Single retrieval ────────

        embedding = self.embed_fn(task)

        if needs_memory:

            memories = self.retrieve_memories(
                query=task,
                embedding=embedding,
                agent_id=agent_id,
                top_k=top_k
            )

        else:

            memories = {
                "query": task,
                "working_memory": [],
                "episodic_memory": [],
                "semantic_memory": [],
                "fused_memory": [],
                "scores": {},
                "skipped": "small_talk" if intent == TaskIntent.SMALL_TALK
                           else "not_required",
            }

        fused = memories.get("fused_memory", [])

        logger.info(
            "[PlannerAgent] "
            "Retrieved: fused=%d relevant memories",
            len(fused)
        )

        # ── Phase 3: Execute ─────────────────

        result = self.executor.execute(
            task=task,
            agent_id=agent_id,
            context=context,
            retrieved_memories=memories,
            embedding=embedding
        )

        # ── Phase 4: Reinforce ───────────────
        #
        # Only after the LLM call succeeded (a failed turn must not
        # count as a successful use of its memories), and only the
        # memories that were actually retrieved — i.e. that passed the
        # relevance cut-off and went into the prompt.
        for mem in fused:
            try:
                # `fused` mixes Episode objects (attribute
                # `episode_id`) and SemanticMemory objects
                # (attribute `memory_id`) — both carry the same
                # underlying id string.
                reinforce_id = getattr(
                    mem, "episode_id", None
                ) or getattr(
                    mem, "memory_id", None
                )

                if reinforce_id is None:
                    continue

                self.reinforce_memory(
                    memory_id=reinforce_id,
                    agent_id=agent_id
                )

                # A memory that got used counts as positive agent
                # feedback (moves Episode.agent_feedback off 0.5).
                self.memory_store.record_feedback(
                    agent_id=agent_id,
                    memory_id=reinforce_id,
                    positive=True
                )

            except Exception as e:
                logger.warning(
                    "[PlannerAgent] "
                    "Reinforce failed: %s", e
                )

        # ── Phase 5: Enrich result ───────────

        result["intent"]          = intent
        result["needs_memory"]    = needs_memory
        result["top_k"]           = top_k
        result["planner_memories"] = memories if needs_memory else {}

        logger.info(
            "[PlannerAgent] "
            "Task complete | intent=%s | "
            "response_len=%d",
            intent,
            len(result.get("response", ""))
        )

        return result