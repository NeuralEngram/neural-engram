# memory/retrieval_config.py

"""
Retrieval tuning knobs, read from the environment at call time
(so tests and a restarted server always see the current value).

RETRIEVAL_MIN_SCORE
    Minimum cosine similarity (0–1) a memory must have with the
    query to be retrieved at all. Below this it is treated as
    irrelevant: not injected into the prompt and not reinforced.
    Default 0.30 suits all-MiniLM-L6-v2 (paraphrases ≈ 0.5–0.8,
    unrelated sentences ≈ 0.0–0.2). Run
    `python -m evaluation.retrieval_eval` to tune it on real
    embeddings.

RETRIEVE_ASSISTANT_REPLIES
    "true" to let the assistant's own past replies be retrieved as
    long-term memories. Default false: only things the user said
    count as facts, so a wrong reply can't come back as "memory".
    Replies are still kept in working memory for conversational
    flow.
"""

import os

DEFAULT_MIN_SCORE = 0.30

# How many memories each intent asks for. SMALL_TALK skips
# retrieval entirely (see agents/planner_agent.py).
TOP_K_BY_INTENT = {
    "memory_retrieval": 8,
    "reasoning": 6,
    "general_response": 4,
    "small_talk": 0,
}

DEFAULT_TOP_K = 5


def min_relevance_score() -> float:
    raw = os.getenv("RETRIEVAL_MIN_SCORE", "").strip()
    if not raw:
        return DEFAULT_MIN_SCORE
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MIN_SCORE
    if value != value:  # NaN
        return DEFAULT_MIN_SCORE
    return max(0.0, min(value, 1.0))


def include_assistant_replies() -> bool:
    return (
        os.getenv("RETRIEVE_ASSISTANT_REPLIES", "false")
        .strip().lower() == "true"
    )


def excluded_roles() -> list:
    return [] if include_assistant_replies() else ["assistant"]


def top_k_for_intent(intent: str) -> int:
    return TOP_K_BY_INTENT.get(intent, DEFAULT_TOP_K)
