"""
MAEM retrieval benchmark — produces evaluation/RETRIEVAL_RESULTS.md.

    python -m evaluation.retrieval_eval                  # real MiniLM embeddings
    python -m evaluation.retrieval_eval --embedder hash  # no model needed (CI)

What it measures
────────────────
A labelled memory set for one user (20 facts the user told the assistant,
plus 3 assistant replies as distractors) and 30 queries:

- 22 answerable questions, each with the gold memory/memories that
  answer it — mostly paraphrased, so lexical overlap alone won't work;
- 8 queries with no relevant memory (off-topic questions + small talk),
  where the right behaviour is to inject nothing.

For each configuration it reports standard retrieval metrics:

  Hit rate      share of answerable queries with ≥1 gold memory retrieved
  Recall        |gold ∩ retrieved| / |gold|, averaged
  Precision     |gold ∩ retrieved| / |retrieved|, averaged over queries
                that retrieved anything
  F1            harmonic mean of the two
  MRR           mean reciprocal rank of the first gold memory
  Noise rejection
                share of no-answer queries that retrieved nothing
  Irrelevant / query
                average number of non-gold memories injected into the
                prompt — the "context noise" the LLM has to ignore
  Latency       mean retrieval time per query

Configurations
──────────────
  Baseline   — the pre-Phase-9 behaviour: fixed top-5, no relevance
               cut-off, assistant replies retrievable, every query
               retrieves (including small talk).
  Phase 9    — intent-routed top-k, relevance cut-off
               (RETRIEVAL_MIN_SCORE), assistant replies excluded, small
               talk skips retrieval.
  + a sweep of the cut-off so you can pick the best value for your
    embedding model.

Runs on a throw-away SQLite file and in-memory Qdrant; never touches the
project database. No API key needed.
"""

import argparse
import hashlib
import os
import re
import statistics
import sys
import tempfile
import time

_TMP = tempfile.mkdtemp(prefix="maem_retrieval_eval_")
os.environ["MAEM_DB_PATH"] = os.path.join(_TMP, "retrieval_eval.db")
os.environ["AGENT_DECAY_MULTIPLIERS"] = ""

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memory.episodic_memory as episodic_module  # noqa: E402
from agents.planner_agent import TaskIntent, detect_intent  # noqa: E402
from memory.memory_orchestrator import MemoryOrchestrator  # noqa: E402
from memory.memory_store import MemoryStore  # noqa: E402
from memory.retrieval_config import (  # noqa: E402
    DEFAULT_MIN_SCORE,
    top_k_for_intent,
)
from memory.semantic_memory import SemanticMemoryStore  # noqa: E402
from memory.working_memory import WorkingMemory  # noqa: E402

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "RETRIEVAL_RESULTS.md"
)
AGENT = "eval_user"

# ─────────────────────────────────────────────────────────────
# Labelled data
# ─────────────────────────────────────────────────────────────

MEMORIES = [
    "My name is Ravi and I am 72 years old",                              # 0
    "I live in Jayanagar, Bangalore with my wife Lakshmi",                # 1
    "My daughter Priya lives in Pune and visits every Sunday",            # 2
    "My son Arjun works as a software engineer in Hyderabad",             # 3
    "I take my blood pressure tablet every morning after breakfast",      # 4
    "I have to take insulin before dinner every day",                     # 5
    "My doctor is Dr. Mehta at Apollo Hospital",                          # 6
    "I have a cardiology appointment next Tuesday at 10 am",              # 7
    "I love gardening, especially growing tomatoes and roses",            # 8
    "I enjoy listening to old Kishore Kumar songs in the evening",        # 9
    "I used to work as a railway station master for 35 years",            # 10
    "My favourite food is masala dosa from Vidyarthi Bhavan",             # 11
    "I walk in Lalbagh park for thirty minutes every morning",            # 12
    "My grandson Rohan is studying engineering and turns 20 in March",    # 13
    "I am allergic to penicillin",                                        # 14
    "Lakshmi makes filter coffee for me every evening",                   # 15
    "I keep my house keys in the drawer next to the front door",          # 16
    "My neighbour Suresh helps me with groceries on Saturdays",           # 17
    "I watch the cricket match whenever India plays",                     # 18
    "My reading glasses are usually on the bedside table",                # 19
]

# The assistant's own past replies — should never be served as "facts".
ASSISTANT_REPLIES = [
    "That sounds lovely, gardening is a very relaxing hobby.",
    "Please remember to take your medicine on time every day.",
    "Cricket is a great sport to watch with the family.",
]

ANSWERABLE = [
    ("What is my name?", {0}),
    ("How old am I?", {0}),
    ("Where do I live?", {1}),
    ("What is my wife's name?", {1, 15}),
    ("Which city does my daughter live in?", {2}),
    ("When does Priya come to visit?", {2}),
    ("What does my son do for a living?", {3}),
    ("When should I take my BP medicine?", {4}),
    ("When do I need my insulin injection?", {5}),
    ("Who is my doctor?", {6}),
    ("When is my heart specialist appointment?", {7}),
    ("What are my hobbies?", {8, 9}),
    ("What kind of music do I like?", {9}),
    ("What was my job before I retired?", {10}),
    ("What is my favourite dish?", {11}),
    ("Where do I go for my morning walk?", {12}),
    ("When is my grandson's birthday?", {13}),
    ("Am I allergic to any medicine?", {14}),
    ("Where did I leave my keys?", {16}),
    ("Who helps me buy groceries?", {17}),
    ("Where are my spectacles?", {19}),
    ("Do I follow any sport?", {18}),
]

NO_ANSWER = [
    "What is the capital of France?",
    "Explain how photosynthesis works",
    "Write a short poem about the ocean",
    "What is 17 times 23?",
    "How do airplanes stay in the air?",
    "Translate good morning into Spanish",
    "hi",
    "ok thanks",
]


# ─────────────────────────────────────────────────────────────
# Embedders
# ─────────────────────────────────────────────────────────────

def hash_embed(text):
    """Deterministic 384-d bag-of-words vector (no model download)."""
    v = [0.0] * 384
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1.0
    if not any(v):
        v[0] = 1.0
    return v


def make_embedder(name):
    if name == "hash":
        return hash_embed, "hash (bag-of-words, no model)"
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover - depends on install
        print(
            f"[retrieval_eval] could not load all-MiniLM-L6-v2 ({exc}); "
            "falling back to --embedder hash"
        )
        return hash_embed, "hash (MiniLM unavailable)"

    cache = {}

    def minilm(text):
        if text not in cache:
            cache[text] = model.encode(text).tolist()
        return cache[text]

    return minilm, "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────

def _assert_throwaway_db():
    """Never wipe a real database: the benchmark clears the episodic DB."""
    db = os.path.realpath(episodic_module.DB_PATH)
    tmp = os.path.realpath(tempfile.gettempdir())
    if not db.startswith(tmp + os.sep):
        raise RuntimeError(
            f"retrieval_eval refuses to run against {db}: it clears the "
            "episodic database. Run it as `python -m evaluation."
            "retrieval_eval` (which uses a temporary DB)."
        )


def build_store(embed):
    _assert_throwaway_db()
    episodic = episodic_module.EpisodicMemoryStore()
    for ep in episodic.get_all():
        episodic.delete(ep.episode_id)
    store = MemoryStore(MemoryOrchestrator(
        WorkingMemory(capacity=100), episodic, SemanticMemoryStore()))
    ids = []
    for text in MEMORIES:
        ids.append(store.store(
            agent_id=AGENT, content=text, embedding=embed(text),
            context={"role": "user"})["memory_id"])
    for text in ASSISTANT_REPLIES:
        store.store(
            agent_id=AGENT, content=text, embedding=embed(text),
            context={"role": "assistant"})
    return store, ids


def mem_id(m):
    return getattr(m, "episode_id", None) or getattr(m, "memory_id", None)


def run_config(store, ids, embed, *, routed, min_score, include_assistant):
    os.environ["RETRIEVE_ASSISTANT_REPLIES"] = (
        "true" if include_assistant else "false")
    index_of = {mid: i for i, mid in enumerate(ids)}

    hits, recalls, precisions, rr, latencies = [], [], [], [], []
    irrelevant, injected, rejected = [], [], 0

    def retrieve(query):
        if routed:
            intent = detect_intent(query)
            if intent == TaskIntent.SMALL_TALK:
                return []
            top_k = top_k_for_intent(intent)
        else:
            top_k = 5
        start = time.perf_counter()
        fused = store.retrieve(
            agent_id=AGENT, query=query, embedding=embed(query),
            top_k=top_k, min_score=min_score)["fused_memory"]
        latencies.append((time.perf_counter() - start) * 1000)
        return fused

    for query, gold in ANSWERABLE:
        fused = retrieve(query)
        got = [index_of.get(mem_id(m), -1) for m in fused]
        found = gold & set(got)
        hits.append(1.0 if found else 0.0)
        recalls.append(len(found) / len(gold))
        if got:
            precisions.append(len(found) / len(got))
        first = next((r for r, g in enumerate(got, 1) if g in gold), None)
        rr.append(1.0 / first if first else 0.0)
        irrelevant.append(len([g for g in got if g not in gold]))
        injected.append(len(got))

    for query in NO_ANSWER:
        fused = retrieve(query)
        if not fused:
            rejected += 1
        irrelevant.append(len(fused))
        injected.append(len(fused))

    p = statistics.mean(precisions) if precisions else 0.0
    r = statistics.mean(recalls)
    # Overall accuracy: a query is handled correctly if an answerable
    # one retrieved ≥1 gold memory, or a no-answer one retrieved nothing.
    correct = sum(hits) + rejected
    return {
        "accuracy": correct / (len(ANSWERABLE) + len(NO_ANSWER)),
        "hit": statistics.mean(hits),
        "recall": r,
        "precision": p,
        "f1": (2 * p * r / (p + r)) if (p + r) else 0.0,
        "mrr": statistics.mean(rr),
        "noise_rejection": rejected / len(NO_ANSWER),
        "irrelevant_per_query": statistics.mean(irrelevant),
        "injected_per_query": statistics.mean(injected),
        "latency_ms": statistics.mean(latencies) if latencies else 0.0,
    }


def count_query_time_embeddings(store, embed):
    """How many stored memories get re-embedded while answering queries."""
    calls = []
    original = episodic_module.simple_embedding
    episodic_module.simple_embedding = (
        lambda text: calls.append(text) or embed(text))
    try:
        for query, _ in ANSWERABLE:
            store.retrieve(agent_id=AGENT, query=query,
                           embedding=embed(query), top_k=5)
    finally:
        episodic_module.simple_embedding = original
    return len(calls) / len(ANSWERABLE)


def pct(x):
    return f"{x * 100:.1f}%"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--embedder", choices=["minilm", "hash"],
                        default="minilm")
    parser.add_argument("--min-score", type=float, default=None,
                        help=f"cut-off for the Phase 9 row "
                             f"(default {DEFAULT_MIN_SCORE})")
    args = parser.parse_args(argv)

    embed, embedder_name = make_embedder(args.embedder)
    phase9_cutoff = (args.min_score if args.min_score is not None
                     else DEFAULT_MIN_SCORE)
    rows, sweep, best_cutoff, reembeds = run_benchmark(embed, phase9_cutoff)
    write_report(args, embedder_name, rows, sweep, best_cutoff, reembeds)

    print(f"Wrote {RESULTS_PATH}")
    for name, m in rows:
        print(f"  {name}: ACCURACY={pct(m['accuracy'])} "
              f"F1={pct(m['f1'])} MRR={m['mrr']:.3f} "
              f"noise-rejection={pct(m['noise_rejection'])} "
              f"irrelevant/query={m['irrelevant_per_query']:.2f}")
    print(f"  best cut-off on this set: {best_cutoff:.2f}")
    return rows, sweep


def assistant_reply_leak_count(store, embed):
    """Queries aimed squarely at the assistant replies; count leaks."""
    os.environ["RETRIEVE_ASSISTANT_REPLIES"] = "false"
    leaks = 0
    for reply in ASSISTANT_REPLIES:
        fused = store.retrieve(agent_id=AGENT, query=reply,
                               embedding=embed(reply), top_k=8)["fused_memory"]
        leaks += sum(1 for m in fused if m.content in ASSISTANT_REPLIES)
    return leaks


def run_benchmark(embed, phase9_cutoff=DEFAULT_MIN_SCORE):
    """Returns (rows, sweep, best_cutoff, reembeds_per_query)."""
    previous = os.environ.get("RETRIEVE_ASSISTANT_REPLIES")
    episodic_module.simple_embedding = embed
    try:
        return _run_benchmark(embed, phase9_cutoff)
    finally:
        if previous is None:
            os.environ.pop("RETRIEVE_ASSISTANT_REPLIES", None)
        else:
            os.environ["RETRIEVE_ASSISTANT_REPLIES"] = previous


def _run_benchmark(embed, phase9_cutoff):
    store, ids = build_store(embed)

    rows = [
        ("Baseline (fixed top-5, no cut-off, replies included)",
         run_config(store, ids, embed, routed=False, min_score=-1.0,
                    include_assistant=True)),
        (f"Phase 9 (intent top-k, cut-off {phase9_cutoff:.2f}, "
         "replies excluded)",
         run_config(store, ids, embed, routed=True, min_score=phase9_cutoff,
                    include_assistant=False)),
    ]

    sweep = []
    for cutoff in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        sweep.append((cutoff, run_config(
            store, ids, embed, routed=True, min_score=cutoff,
            include_assistant=False)))
    best_cutoff, best = max(
        sweep, key=lambda item: (
            round(item[1]["f1"] + item[1]["noise_rejection"], 6),
            item[1]["recall"]))

    reembeds = count_query_time_embeddings(store, embed)
    leaks = assistant_reply_leak_count(store, embed)
    rows[1][1]["assistant_leaks"] = leaks
    return rows, sweep, best_cutoff, reembeds


def write_report(args, embedder_name, rows, sweep, best_cutoff, reembeds):
    header = ("| Configuration | Accuracy | Hit rate | Recall | Precision | "
              "F1 | MRR | Noise rejection | Irrelevant / query | "
              "Injected / query | Latency |")
    sep = "|" + "---|" * 11

    def line(name, m):
        return (f"| {name} | **{pct(m['accuracy'])}** | {pct(m['hit'])} | "
                f"{pct(m['recall'])} | "
                f"{pct(m['precision'])} | {pct(m['f1'])} | {m['mrr']:.3f} | "
                f"{pct(m['noise_rejection'])} | "
                f"{m['irrelevant_per_query']:.2f} | "
                f"{m['injected_per_query']:.2f} | "
                f"{m['latency_ms']:.1f} ms |")

    out = [
        "# Retrieval benchmark",
        "",
        "Generated by `python -m evaluation.retrieval_eval"
        + (" --embedder hash" if args.embedder == "hash" else "") + "`.",
        "",
        f"- Embedding model: **{embedder_name}**",
        f"- Memory set: {len(MEMORIES)} user facts + "
        f"{len(ASSISTANT_REPLIES)} assistant replies (distractors)",
        f"- Queries: {len(ANSWERABLE)} answerable (labelled gold memories), "
        f"{len(NO_ANSWER)} with no relevant memory (off-topic + small talk)",
        "",
        "## Baseline vs Phase 9",
        "",
        header, sep,
        *[line(name, m) for name, m in rows],
        "",
        "## Relevance cut-off sweep (Phase 9 pipeline)",
        "",
        header, sep,
        *[line(f"cut-off {c:.2f}", m) for c, m in sweep],
        "",
        f"Best trade-off on this set (F1 + noise rejection): "
        f"**{best_cutoff:.2f}** — set `RETRIEVAL_MIN_SCORE={best_cutoff:.2f}` "
        f"in `.env` to use it.",
        "",
        "## Efficiency",
        "",
        f"- Stored memories re-embedded per query: **{reembeds:.1f}** "
        f"(before Phase 9 the episodic tier re-embedded every stored memory "
        f"on every query — {len(MEMORIES) + len(ASSISTANT_REPLIES)} here, "
        f"growing linearly with memory size).",
        "- Retrievals per chat turn: **1** (before: 2 — planner and executor "
        "each retrieved).",
        "- Task embeddings per chat turn: **1** (before: 2).",
        "- Small talk: **0** retrievals.",
        "",
        "## Metric definitions",
        "",
        "- **Accuracy** — share of all queries handled correctly: an "
        "answerable query retrieved at least one gold memory, or a "
        "no-answer query retrieved nothing.",
        "- **Hit rate** — answerable queries where at least one gold memory "
        "was retrieved.",
        "- **Recall** — share of each query's gold memories retrieved.",
        "- **Precision** — share of retrieved memories that were gold "
        "(queries that retrieved nothing are excluded).",
        "- **MRR** — 1 / rank of the first gold memory (0 if none).",
        "- **Noise rejection** — no-answer queries for which nothing was "
        "injected.",
        "- **Irrelevant / query** — non-gold memories put into the prompt, "
        "averaged over all queries.",
    ]
    if args.embedder == "hash" or "unavailable" in embedder_name:
        out += [
            "",
            "> ⚠️ Run with the hash embedder: it only matches identical "
            "words, so paraphrased questions score poorly. Run "
            "`python -m evaluation.retrieval_eval` with MiniLM installed "
            "for representative numbers.",
        ]

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
