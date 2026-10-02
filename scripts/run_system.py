#!/usr/bin/env python3

# run_system.py

"""
NeuralEngram — Live Conversation Loop

Usage
─────
    python scripts/run_system.py

Commands
────────
    /quit   or   /exit     → exit the system
    /memory                → show current memory metrics
    /context               → show working memory context
    /clear                 → apply decay + cleanup forgotten memories
    /intent <query>        → dry-run intent detection (no execution)
    /help                  → show this help

Notes
─────
- All conversations are persisted to memory automatically.
- Memories decay over time via Ebbinghaus curve (every 5 min).
- Type naturally — the planner decides if memory retrieval is needed.
"""

import logging
import sys
import os

# ── Allow running from project root ──────────────────────────
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from orchestrator.orchestrator import Orchestrator
from agents.planner_agent import TaskIntent
from memory.retrieval_config import top_k_for_intent


# ─────────────────────────────────────────────────────────────
# Logging — only WARNING+ for clean terminal output
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

INTENT_LABELS = {
    TaskIntent.MEMORY:    "🧠 memory retrieval",
    TaskIntent.REASONING: "🔍 reasoning",
    TaskIntent.GENERAL:   "💬 general response",
    TaskIntent.SMALL_TALK: "👋 small talk (no retrieval)",
}


def print_header():
    print()
    print("█" * 62)
    print("  NEURAALENGRAM — Multi-Agent AI System")
    print("  Live Conversation Mode")
    print("  Team: TimeOut (Anika · Azmat · Poojitha)")
    print("█" * 62)
    print()
    print("  Type your message and press Enter.")
    print("  Type /help for commands.")
    print()


def print_response(result: dict):
    intent = result.get("intent", "unknown")
    label  = INTENT_LABELS.get(intent, intent)

    fused  = result.get(
        "planner_memories", {}
    ).get("fused_memory", [])

    mem_count = len(fused)

    print()
    print("─" * 62)
    print(f"  Intent  : {label}")

    if mem_count > 0:
        print(f"  Memories: {mem_count} retrieved")

    print("─" * 62)
    print()
    print(result["response"])
    print()


def cmd_memory(orchestrator: Orchestrator):
    metrics = orchestrator.planner.memory_metrics()
    print()
    print("─" * 62)
    print("  MEMORY METRICS")
    print("─" * 62)
    print(metrics)
    print()


def cmd_context(orchestrator: Orchestrator):
    context = orchestrator.planner.get_context(limit=10)
    print()
    print("─" * 62)
    print("  WORKING MEMORY CONTEXT")
    print("─" * 62)
    if not context:
        print("  (empty)")
    for i, msg in enumerate(context, 1):
        role    = msg.get("role", "?")
        content = msg.get("content", "")[:120]
        print(f"  [{i}] {role}: {content}")
    print()


def cmd_clear(orchestrator: Orchestrator):
    orchestrator.planner.apply_decay()
    removed = orchestrator.planner.cleanup_memories()
    print()
    print(f"  ✓ Decay applied. {removed} memories cleaned up.")
    print()


def cmd_intent(orchestrator: Orchestrator, query: str):
    if not query.strip():
        print("  Usage: /intent <your query here>")
        return
    intent = orchestrator.planner._detect_intent(query)
    label  = INTENT_LABELS.get(intent, intent)
    needs  = orchestrator.planner._requires_memory(intent)
    top_k  = top_k_for_intent(intent) if needs else 0
    print()
    print(f"  Intent       : {label}")
    print(f"  Needs memory : {needs}")
    print(f"  Top-k        : {top_k}")
    print()


def cmd_help():
    print()
    print("─" * 62)
    print("  COMMANDS")
    print("─" * 62)
    print("  /quit   /exit    → exit the system")
    print("  /memory          → show memory metrics")
    print("  /context         → show working memory context")
    print("  /clear           → decay + cleanup memories")
    print("  /intent <query>  → dry-run intent detection")
    print("  /help            → show this help")
    print()


# ─────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────

def main():

    print_header()

    print("  Initializing NeuralEngram system...")
    print("  (Loading embedding model — first run may take ~10s)")
    print()

    try:
        orchestrator = Orchestrator()
    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        sys.exit(1)

    print("  ✓ System ready.\n")

    try:
        while True:

            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Goodbye.\n")
                break

            if not user_input:
                continue

            # ── Commands ─────────────────────────────────────

            if user_input.lower() in ("/quit", "/exit"):
                print("\n  Goodbye.\n")
                break

            elif user_input.lower() == "/memory":
                cmd_memory(orchestrator)
                continue

            elif user_input.lower() == "/context":
                cmd_context(orchestrator)
                continue

            elif user_input.lower() == "/clear":
                cmd_clear(orchestrator)
                continue

            elif user_input.lower().startswith("/intent "):
                query = user_input[8:]
                cmd_intent(orchestrator, query)
                continue

            elif user_input.lower() == "/help":
                cmd_help()
                continue

            elif user_input.startswith("/"):
                print(
                    f"  Unknown command: {user_input}. "
                    f"Type /help for commands."
                )
                continue

            # ── Route through PlannerAgent ────────────────────

            try:
                result = orchestrator.route(user_input)
                print_response(result)

            except Exception as e:
                print(
                    f"\n  ✗ Error: {e}\n"
                    f"  Please try again.\n"
                )
                logger.error(
                    "Route error: %s", e,
                    exc_info=True
                )

    finally:
        orchestrator.stop()


if __name__ == "__main__":
    main()