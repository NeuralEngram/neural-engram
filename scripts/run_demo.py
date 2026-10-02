"""
Demo Session Runner — runs a live session through the multi-agent system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.orchestrator import Orchestrator

def run_demo():
    print("=== MultiAgent Memory System — Demo Session ===\n")

    orch = Orchestrator()

    # Seed memories with embeddings
    memories = [
        "Ebbinghaus forgetting curve reduces retention over time",
        "Spaced repetition reinforces memory strength",
        "MemGPT uses external memory for long context",
    ]

    for mem in memories:
        orch.executor.store_memory(
            content=mem,
            embedding=orch._embed(mem),
            context={"role": "user"}
        )

    print("✅ Memory seeded\n")

    # Run sessions through orchestrator
    sessions = [
        "What do we know about Ebbinghaus forgetting curve?",
        "How does spaced repetition strengthen memory?",
        "Explain MemGPT external memory",
    ]

    for prompt in sessions:
        print(f"🧠 Prompt: {prompt}")
        result = orch.route(prompt)
        print(f"   Response: {result['response'][:150]}")
        print()

    orch.stop()
    print("✅ Demo complete!")

if __name__ == "__main__":
    run_demo()