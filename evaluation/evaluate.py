"""
MAEM internal validation — produces evaluation/RESULTS.md.

    python -m evaluation.evaluate            # from the project root

Measures
────────
1. Retention / decay accuracy — EbbinghausEngine.compute_retention() and
   the full Episode.retention() path vs. the closed form R = e^(-t/S).
2. Caregiver-alert precision / recall — detect_alerts() on a labelled
   synthetic set, scored three ways (rule conformance, caregiver intent,
   and end-to-end through a real scheduler maintenance cycle).
3. Feedback-loop convergence (Phase 1) — agent_feedback /
   task_success_rate under repeated and noisy signals through the real
   MemoryStore → MemoryOrchestrator → EpisodicMemoryStore (SQLite).
4. Retrieval quality & efficiency (Phase 9) — the labelled benchmark in
   evaluation/retrieval_eval.py (precision, recall, MRR, noise
   rejection, context noise, re-embeddings per query), baseline vs the
   Phase 9 pipeline. Uses the deterministic bag-of-words embedder here;
   run `python -m evaluation.retrieval_eval` for real-model numbers.

Reproducibility
───────────────
- Fixed clock for every retention computation; seeded RNG for noisy
  signals. Rerunning produces identical numbers.
- Runs on a throw-away SQLite file and in-memory Qdrant. It never touches
  the project database or qdrant_data/.
- Needs no model download and no API key.

Exit status is non-zero if any correctness gate fails, so this doubles as
a regression check.
"""

import hashlib
import math
import os
import random
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone

# ── Isolation: MUST happen before importing memory.episodic_memory,
#    which resolves its DB path at import time. ──────────────────────
_TMP = tempfile.mkdtemp(prefix="maem_eval_")
os.environ["MAEM_DB_PATH"] = os.path.join(_TMP, "eval.db")
os.environ["AGENT_DECAY_MULTIPLIERS"] = "fast_agent:3.0"
# never send real email/SMS from an evaluation run
os.environ["ENABLE_EMAIL_ALERTS"] = "false"
os.environ["ENABLE_SMS_ALERTS"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.ebbinghaus import (  # noqa: E402
    DEFAULT_RETENTION_THRESHOLD,
    MAX_STABILITY_HOURS,
    EbbinghausEngine,
    compute_retention,
    time_until_forgotten,
)
from memory.episodic_memory import Episode, EpisodicMemoryStore  # noqa: E402
from memory.memory_orchestrator import MemoryOrchestrator  # noqa: E402
from memory.memory_store import MemoryStore  # noqa: E402
from memory.notifier import CRITICAL_KEYWORDS, CaregiverNotifier, detect_alerts  # noqa: E402
from memory.scheduler import MemoryScheduler  # noqa: E402
from memory.semantic_memory import SemanticMemoryStore  # noqa: E402
from memory.working_memory import WorkingMemory  # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "RESULTS.md")
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)  # fixed clock
SEED = 20260925
ALPHA = 0.3  # EMA step used by record_feedback / record_task_outcome
GATES = []   # (name, passed, detail)


def gate(name, passed, detail):
    GATES.append((name, bool(passed), detail))


def embed(text):
    """Deterministic 384-d bag-of-words vector (no model download)."""
    v = [0.0] * 384
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1.0
    if not any(v):
        v[0] = 1.0
    return v


def fresh_stack():
    episodic = EpisodicMemoryStore()
    store = MemoryStore(MemoryOrchestrator(
        WorkingMemory(), episodic, SemanticMemoryStore()))
    return store, episodic


def reset_db():
    store = EpisodicMemoryStore()
    for ep in store.get_all():
        store._delete_episode(ep.episode_id)


# ═════════════════════════════════════════════════════════════
# 1. Retention / decay accuracy
# ═════════════════════════════════════════════════════════════

STABILITIES = [1, 6, 24, 31.2, 45.6, 60, 72, 168, 720, 8760]
ELAPSED = [0, 0.5, 1, 6, 12, 24, 48, 72, 168, 720, 2000, 8760]


def eval_retention():
    engine = EbbinghausEngine()
    out = {}

    # 1a. engine vs closed form over the grid (S within the cap)
    errs = []
    for S in STABILITIES:
        for t in ELAPSED:
            errs.append(abs(engine.compute_retention(t, S) - math.exp(-t / S)))
    out["grid_points"] = len(errs)
    out["grid_max_err"] = max(errs)
    out["grid_mean_err"] = sum(errs) / len(errs)
    gate("retention: engine == e^(-t/S) on grid", out["grid_max_err"] < 1e-12,
         f"max |err| = {out['grid_max_err']:.2e} over {len(errs)} points")

    # 1b. characteristic points
    checks = []
    for S in (24, 45.6, 72):
        checks.append(("R(0)", S, engine.compute_retention(0, S), 1.0))
        checks.append(("R(S) = 1/e", S, engine.compute_retention(S, S),
                       math.exp(-1)))
        checks.append(("R(S·ln2) = 0.5", S,
                       engine.compute_retention(S * math.log(2), S), 0.5))
        checks.append(("R(S·ln(1/θ)) = θ", S, engine.compute_retention(
            S * math.log(1 / DEFAULT_RETENTION_THRESHOLD), S),
            DEFAULT_RETENTION_THRESHOLD))
    out["characteristic"] = checks
    char_err = max(abs(a - b) for _, _, a, b in checks)
    gate("retention: characteristic points", char_err < 1e-12,
         f"max |err| = {char_err:.2e} over {len(checks)} checks")

    # 1c. documented clamps
    out["negative_t"] = engine.compute_retention(-5, 24)
    capped = [(S, t, engine.compute_retention(t, S), math.exp(-t / S),
               math.exp(-t / MAX_STABILITY_HOURS))
              for S in (10_000, 17_520) for t in (720, 8760)]
    out["capped"] = capped
    gate("retention: negative elapsed clamps to 1.0",
         out["negative_t"] == 1.0, f"R(-5h) = {out['negative_t']}")
    gate("retention: S above cap behaves as S = cap",
         all(abs(r - cap) < 1e-12 for _, _, r, _, cap in capped),
         f"{len(capped)} points")

    # 1d. full Episode path (timestamps + per-agent decay multiplier)
    ep_errs = []
    for mult in (1.0, 0.5, 3.0):
        for S in (31.2, 45.6, 60):
            for t in (1, 12, 24, 48, 96):
                ep = Episode(content="x", context={}, agent_id="a",
                             stability_hours=S, decay_multiplier=mult,
                             last_reviewed_at=T0)
                got = ep.retention(T0 + timedelta(hours=t))
                ep_errs.append(abs(got - math.exp(-t * mult / S)))
    out["episode_points"] = len(ep_errs)
    out["episode_max_err"] = max(ep_errs)
    gate("retention: Episode.retention == e^(-t·m/S)",
         out["episode_max_err"] < 1e-9,
         f"max |err| = {out['episode_max_err']:.2e} over {len(ep_errs)} "
         "points (float timestamps)")

    # 1e. forgetting boundary via is_forgotten()
    boundary = []
    for S in (31.2, 45.6, 60):
        t_star = S * math.log(1 / DEFAULT_RETENTION_THRESHOLD)
        ep = Episode(content="x", context={}, agent_id="a",
                     stability_hours=S, last_reviewed_at=T0)
        before = ep.is_forgotten(T0 + timedelta(hours=t_star - 0.01))
        after = ep.is_forgotten(T0 + timedelta(hours=t_star + 0.01))
        predicted = time_until_forgotten(T0, S, now=T0)
        boundary.append((S, t_star, before, after, predicted))
    out["boundary"] = boundary
    gate("retention: is_forgotten flips at S·ln(1/θ) ± 36 s",
         all((not b) and a for _, _, b, a, _ in boundary),
         f"{len(boundary)} stabilities")
    gate("retention: time_until_forgotten() agrees",
         all(abs(p - t) < 1e-9 for _, t, _, _, p in boundary), "")

    # 1f. what this means for real memories (stability from importance)
    store, episodic = fresh_stack()
    samples = {
        "low importance (\"thanks, okay\")": "thanks okay",
        "neutral (\"the garden has tomatoes\")": "the garden has tomatoes",
        "high importance (\"I need my glasses\")": "I need my glasses",
    }
    lifetimes = []
    for label, text in samples.items():
        for agent in ("normal_agent", "fast_agent"):
            mid = store.store(agent, text, embed(text))["memory_id"]
            ep = episodic.get_by_id(mid)
            S_eff = ep.stability_hours / ep.decay_multiplier
            lifetimes.append((label, agent, ep.importance, ep.stability_hours,
                              ep.decay_multiplier,
                              S_eff * math.log(1 / DEFAULT_RETENTION_THRESHOLD)))
    out["lifetimes"] = lifetimes

    # 1g. reinforcement growth through EpisodicMemoryStore.recall()
    mid = store.store("normal_agent", "the garden has tomatoes",
                      embed("the garden has tomatoes"))["memory_id"]
    ep = episodic.get_by_id(mid)
    growth = [(0, ep.stability_hours)]
    for k in range(1, 9):
        episodic.recall(mid)
        growth.append((k, ep.stability_hours))
    out["growth"] = growth
    gate("retention: every recall increases stability",
         all(b[1] > a[1] for a, b in zip(growth, growth[1:])), "8 recalls")
    reset_db()
    return out


# ═════════════════════════════════════════════════════════════
# 2. Alert precision / recall
# ═════════════════════════════════════════════════════════════
#
# target_R : retention the memory is aged to
# intent   : should a caregiver be alerted? (human judgement, written
#            BEFORE running detect_alerts — deliberately includes cases the
#            keyword rule can't get right)

ALERT_SET = [
    # content,                                    target_R, intent, note
    ("my daughter Priya brings my medicine",           0.20, True,  ""),
    ("doctor appointment on friday at 10am",           0.40, True,  ""),
    ("insulin before breakfast every day",             0.25, True,  ""),
    ("emergency contact is Arjun",                     0.45, True,  ""),
    ("blood pressure tablet at night",                 0.35, True,  ""),
    ("grandson Rohan's birthday is in May",            0.40, True,  "hits via 'son'"),
    ("take two pills after lunch",                     0.20, True,  "no keyword"),
    ("the nurse visits on tuesdays",                   0.40, True,  "no keyword"),
    ("wife Kamala's phone number is 98450",            0.25, True,  "no keyword"),
    ("hospital visit next month for a checkup",        0.90, False, "still fresh"),
    ("my family lives in Pune",                        0.60, False, "retention OK"),
    ("doctor said to walk daily",                      0.52, False, "just above 0.5"),
    ("I enjoy painting landscapes by the lake",        0.20, False, ""),
    ("favourite film is Sholay",                       0.10, False, ""),
    ("the weather was pleasant yesterday",             0.30, False, ""),
    ("remember to lock the front door",                0.30, False, ""),
    ("met a nice person at the park",                  0.20, False, "'son' in person"),
    ("autumn is my favourite season",                  0.20, False, "'son' in season"),
    ("helped with the grandkids' homework",            0.35, False, "'home' in homework"),
    ("the reason I moved here was the climate",        0.20, False, "'son' in reason"),
]
AGENT = "eval_patient"


def spec_rule(content, R):
    """The documented rule, re-implemented independently."""
    hit = any(k in content.lower() for k in CRITICAL_KEYWORDS)
    if not hit or R >= 0.5:
        return None
    return "critical" if R < 0.3 else "warning"


def build_alert_episodes(episodic, store, now):
    by_content = {}
    for content, R, _, _ in ALERT_SET:
        mid = store.store(AGENT, content, embed(content))["memory_id"]
        ep = episodic.get_by_id(mid)
        hours = -math.log(R) * ep.stability_hours / ep.decay_multiplier
        ep.last_reviewed_at = now - timedelta(hours=hours)
        episodic.save(ep)
        by_content[content] = ep
    return by_content


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def eval_alerts():
    out = {}
    store, episodic = fresh_stack()
    now = datetime.now(timezone.utc)
    build_alert_episodes(episodic, store, now)

    raised = {a["memory"]: a["alert_level"]
              for a in detect_alerts(episodic.get_all(), AGENT)}
    rows = []
    for content, R, intent, note in ALERT_SET:
        rows.append((content, R, spec_rule(content, R), raised.get(content),
                     intent, note))
    out["rows"] = rows

    # (a) implementation vs documented rule
    agree = sum(1 for _, _, spec, got, _, _ in rows if spec == got)
    tp = sum(1 for _, _, s, g, _, _ in rows if s and g)
    fp = sum(1 for _, _, s, g, _, _ in rows if g and not s)
    fn = sum(1 for _, _, s, g, _, _ in rows if s and not g)
    out["rule"] = (agree, len(rows), *prf(tp, fp, fn))
    gate("alerts: detect_alerts matches documented rule (incl. level)",
         agree == len(rows), f"{agree}/{len(rows)}")

    # (b) vs caregiver intent
    tp = sum(1 for r in rows if r[3] and r[4])
    fp = sum(1 for r in rows if r[3] and not r[4])
    fn = sum(1 for r in rows if not r[3] and r[4])
    tn = sum(1 for r in rows if not r[3] and not r[4])
    out["intent"] = (tp, fp, fn, tn, *prf(tp, fp, fn))

    # (c) end to end: one real scheduler maintenance cycle, alert hook
    #     wired exactly like Orchestrator.check_caregiver_alerts
    reset_db()
    store, episodic = fresh_stack()
    build_alert_episodes(episodic, store, now)
    notifier = CaregiverNotifier()

    def hook(_event):
        notifier.notify(AGENT, detect_alerts(episodic.get_all(), AGENT))
    sched = MemoryScheduler(store, interval_seconds=300,
                            on_cycle_complete=hook)
    sched.run_once()
    notified = {m for (_, m, _) in notifier._already_notified}
    tp = sum(1 for r in rows if r[4] and r[0] in notified)
    fp = sum(1 for r in rows if r[0] in notified and not r[4])
    fn = sum(1 for r in rows if r[4] and r[0] not in notified)
    lost_critical = [r[0] for r in rows if r[2] == "critical"
                     and r[0] not in notified]
    out["pipeline"] = (tp, fp, fn, *prf(tp, fp, fn), lost_critical,
                       sched.total_failures)
    reset_db()
    return out


# ═════════════════════════════════════════════════════════════
# 3. Feedback-loop convergence (Phase 1)
# ═════════════════════════════════════════════════════════════

def eval_feedback():
    out = {}
    store, episodic = fresh_stack()

    def new_memory(text):
        return store.store("fb_agent", text, embed(text))["memory_id"]

    # 3a. deterministic sequences vs closed form
    seqs = {}
    for field, record, kw in (
        ("agent_feedback", store.record_feedback, "positive"),
        ("task_success_rate", store.record_task_outcome, "success"),
    ):
        for signal in (True, False):
            mid = new_memory(f"{field} {signal}")
            ep = episodic.get_by_id(mid)
            vals, errs = [], []
            for n in range(1, 51):
                record(agent_id="fb_agent", memory_id=mid, **{kw: signal})
                v = getattr(ep, field)
                exp = (1 - 0.5 * (1 - ALPHA) ** n) if signal \
                    else 0.5 * (1 - ALPHA) ** n
                vals.append(v)
                errs.append(abs(v - exp))
            target = 1.0 if signal else 0.0
            steps_to = {
                thr: next(n for n, v in enumerate(vals, 1)
                          if abs(v - target) <= thr)
                for thr in (0.1, 0.01, 0.001)
            }
            monotone = all((b >= a) if signal else (b <= a)
                           for a, b in zip(vals, vals[1:]))
            reloaded = getattr(EpisodicMemoryStore().get_by_id(mid), field)
            seqs[(field, signal)] = dict(
                max_err=max(errs), final=vals[-1], steps=steps_to,
                monotone=monotone, bounded=all(0 <= v <= 1 for v in vals),
                persisted=abs(reloaded - vals[-1]) < 1e-12)
            gate(f"feedback: {field} {'+' if signal else '-'} follows EMA "
                 "closed form", max(errs) < 1e-9 and monotone,
                 f"max |err| = {max(errs):.1e}")
            gate(f"feedback: {field} {'+' if signal else '-'} persisted",
                 seqs[(field, signal)]["persisted"], "")
    out["sequences"] = seqs

    # 3b. noisy signals: does the EMA settle around the true rate p?
    rng = random.Random(SEED)
    noisy = []
    for field, record, kw in (
        ("agent_feedback", store.record_feedback, "positive"),
        ("task_success_rate", store.record_task_outcome, "success"),
    ):
        for p in (0.2, 0.5, 0.8):
            mid = new_memory(f"noisy {field} {p}")
            ep = episodic.get_by_id(mid)
            vals, sent = [], []
            for _ in range(600):
                s_ = rng.random() < p
                sent.append(s_)
                record(agent_id="fb_agent", memory_id=mid, **{kw: s_})
                vals.append(getattr(ep, field))
            tail = vals[100:]  # after burn-in
            observed = sum(sent[100:]) / len(sent[100:])
            mean = sum(tail) / len(tail)
            sd = (sum((v - mean) ** 2 for v in tail) / len(tail)) ** 0.5
            theory_sd = math.sqrt(ALPHA / (2 - ALPHA) * p * (1 - p))
            noisy.append((field, p, observed, mean, sd, theory_sd))
            # Reference = rate of signals actually sent. With 500 draws
            # the observed rate itself varies ±0.022 (1 SE) around p, so
            # comparing to nominal p would test the RNG, not the EMA.
            gate(f"feedback: {field} tracks signal rate (p={p})",
                 abs(mean - observed) < 0.01,
                 f"EMA mean {mean:.3f} vs sent {observed:.3f}")
    out["noisy"] = noisy

    # 3c. alternating +/−: bounded 2-cycle, no drift
    mid = new_memory("alternating")
    ep = episodic.get_by_id(mid)
    vals = []
    for i in range(200):
        store.record_feedback(agent_id="fb_agent", memory_id=mid,
                              positive=(i % 2 == 0))
        vals.append(ep.agent_feedback)
    hi = ALPHA / (1 - (1 - ALPHA) ** 2)
    out["alternating"] = (vals[-2], vals[-1], hi, (1 - ALPHA) * hi)
    gate("feedback: alternating signals settle into fixed 2-cycle",
         abs(vals[-2] - hi) < 1e-9 and abs(vals[-1] - (1 - ALPHA) * hi) < 1e-9,
         f"{vals[-2]:.4f} / {vals[-1]:.4f}")

    # 3d. effect on the DMS priority score
    a, b = new_memory("priority a"), new_memory("priority b")
    ea, eb = episodic.get_by_id(a), episodic.get_by_id(b)
    for e in (ea, eb):
        e.last_reviewed_at = T0
        e.importance = 0.45
    base = ea.priority_score(T0)
    for _ in range(20):
        store.record_feedback(agent_id="fb_agent", memory_id=a, positive=True)
        store.record_task_outcome(agent_id="fb_agent", memory_id=a,
                                  success=True)
        store.record_feedback(agent_id="fb_agent", memory_id=b, positive=False)
        store.record_task_outcome(agent_id="fb_agent", memory_id=b,
                                  success=False)
    out["priority"] = (base, ea.priority_score(T0), eb.priority_score(T0))
    reset_db()
    return out


# ═════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════

def fmt(x, d=4):
    return f"{x:.{d}f}"


def write_report(ret, alr, fb, rq=None):
    L = []
    w = L.append
    passed = sum(1 for _, ok, _ in GATES if ok)
    w("# MAEM — internal validation results\n")
    w("Generated by `python -m evaluation.evaluate` (run from the project "
      "root). The script is deterministic — fixed clock, seeded RNG "
      f"(seed {SEED}) — so rerunning reproduces every number below. It uses "
      "a throw-away SQLite file and in-memory Qdrant, needs no model "
      "download and no API key, and exits non-zero if any gate fails.\n")
    w(f"**Correctness gates: {passed}/{len(GATES)} passed.**\n")

    # 1 ────────────────────────────────────────────────────────
    w("## 1. Retention / decay accuracy\n")
    w("**Method.** `EbbinghausEngine.compute_retention(t, S)` compared with "
      f"the closed form R = e^(−t/S) over a grid of {len(STABILITIES)} "
      f"stabilities × {len(ELAPSED)} elapsed times "
      f"(S ∈ {STABILITIES} h; t ∈ {ELAPSED} h). Then the full production "
      "path, `Episode.retention(now)` (datetime timestamps + per-agent decay "
      "multiplier m), against e^(−t·m/S).\n")
    w("| Check | Points | Max abs. error |")
    w("|---|---|---|")
    w(f"| Engine vs e^(−t/S) | {ret['grid_points']} | "
      f"{ret['grid_max_err']:.1e} |")
    w(f"| Episode.retention vs e^(−t·m/S), m ∈ {{0.5, 1, 3}} | "
      f"{ret['episode_points']} | {ret['episode_max_err']:.1e} |")
    w(f"| Characteristic points (below) | {len(ret['characteristic'])} | "
      f"{max(abs(a - b) for *_, a, b in ret['characteristic']):.1e} |\n")
    w("Characteristic points (S = 45.6 h shown; 24 h and 72 h identical):\n")
    w("| Identity | Engine | Theory |")
    w("|---|---|---|")
    for name, S, got, exp in ret["characteristic"]:
        if S == 45.6:
            w(f"| {name} | {fmt(got, 6)} | {fmt(exp, 6)} |")
    w("")
    w(f"Forgetting threshold θ = {DEFAULT_RETENTION_THRESHOLD}, so a memory "
      f"is forgotten at t* = S·ln(1/θ) ≈ "
      f"{math.log(1 / DEFAULT_RETENTION_THRESHOLD):.3f}·S. `is_forgotten()` "
      "flips within ±0.01 h of t* for every tested S, and "
      "`time_until_forgotten()` predicts t* exactly.\n")
    w("**Documented clamps (by design, not errors).** Negative elapsed time "
      f"returns R = {ret['negative_t']}. Stability above "
      f"MAX_STABILITY_HOURS = {MAX_STABILITY_HOURS:.0f} h is treated as the "
      "cap:\n")
    w("| S (h) | t (h) | Engine | Pure e^(−t/S) | e^(−t/cap) |")
    w("|---|---|---|---|---|")
    for S, t, got, pure, cap in ret["capped"]:
        w(f"| {S} | {t} | {fmt(got)} | {fmt(pure)} | {fmt(cap)} |")
    w("")
    starts = sorted({round(S, 1) for _, _, _, S, _, _ in ret["lifetimes"]})
    w(f"*Practical note:* `add()` gives new memories S between "
      f"{starts[0]} h and {starts[-1]} h, and each `recall()` multiplies S "
      "by ~1.5–1.7 up to the same 8760 h cap, so S itself never exceeds the "
      "cap. The clamp only changes results for an agent with a decay "
      "multiplier < 1 once a memory has been recalled enough for S/m to "
      "pass 8760 h.\n")
    w("**What it means for real memories** (stability assigned by "
      "`EpisodicMemoryStore.add()` = 24 h × (1 + 2·importance); "
      "`fast_agent` configured with multiplier 3.0):\n")
    w("| Memory | Agent | Importance | Stability (h) | Multiplier | "
      "Forgotten after (h, no recall) |")
    w("|---|---|---|---|---|---|")
    for label, agent, imp, S, m, life in ret["lifetimes"]:
        w(f"| {label} | {agent} | {imp} | {S:.1f} | {m} | {life:.1f} |")
    w("")
    w("**Reinforcement** — stability after k recalls through "
      "`EpisodicMemoryStore.recall()` (neutral memory, 45.6 h start):\n")
    w("| Recalls | " + " | ".join(str(k) for k, _ in ret["growth"]) + " |")
    w("|---" * (len(ret["growth"]) + 1) + "|")
    w("| Stability (h) | " + " | ".join(f"{s:.1f}" for _, s in ret["growth"])
      + " |")
    w("| Growth × | " + " | ".join(
        "—" if k == 0 else f"{s / ret['growth'][k - 1][1]:.3f}"
        for k, s in ret["growth"]) + " |\n")

    # 2 ────────────────────────────────────────────────────────
    w("## 2. Caregiver alert precision / recall\n")
    w(f"**Method.** {len(ALERT_SET)} synthetic memories stored through the "
      "real `MemoryStore`, each aged to a target retention by moving "
      "`last_reviewed_at` back (the same mutation `/simulate_time` "
      "performs), then scored with `detect_alerts()`. Each memory carries "
      "a caregiver-intent label written before running the detector: "
      "\"should a caregiver be alerted about this fading memory?\"\n")
    w("Scored three ways:\n")
    w("- **(a) Rule conformance:** does `detect_alerts` implement its "
      "documented rule (critical keyword AND R < 0.5; critical if R < 0.3, "
      "else warning)? The rule is re-implemented independently in the "
      "script as the reference.")
    w("- **(b) Caregiver intent:** how well that rule matches what a "
      "caregiver actually wants.")
    w("- **(c) End-to-end:** one real `MemoryScheduler` maintenance cycle "
      "with the caregiver hook wired as in "
      "`Orchestrator.check_caregiver_alerts`; counts what was actually "
      "notified.\n")
    agree, n, rp, rr, rf = alr["rule"]
    tp, fp, fn, tn, p, r, f = alr["intent"]
    ptp, pfp, pfn, pp, pr, pf, lost, sched_fail = alr["pipeline"]
    w("| Scoring | TP | FP | FN | Precision | Recall | F1 |")
    w("|---|---|---|---|---|---|---|")
    w(f"| (a) vs documented rule | — | — | — | {fmt(rp, 2)} | {fmt(rr, 2)} "
      f"| {fmt(rf, 2)} |")
    w(f"| (b) vs caregiver intent | {tp} | {fp} | {fn} | {fmt(p, 2)} | "
      f"{fmt(r, 2)} | {fmt(f, 2)} |")
    w(f"| (c) end-to-end via scheduler | {ptp} | {pfp} | {pfn} | "
      f"{fmt(pp, 2)} | {fmt(pr, 2)} | {fmt(pf, 2)} |\n")
    w(f"(a) Level agreement (none / warning / critical): {agree}/{n}.\n")
    w("Per-memory detail:\n")
    w("| Memory | Aged to R | Rule expects | detect_alerts | Intent | Note |")
    w("|---|---|---|---|---|---|")
    for content, R, spec, got, intent, note in alr["rows"]:
        w(f"| {content} | {R} | {spec or '—'} | {got or '—'} | "
          f"{'alert' if intent else '—'} | {note} |")
    w("")
    w("**Reading the numbers.**\n")
    w(f"- The implementation is exactly its spec ({agree}/{n}).")
    w(f"- Against intent, the {fp} false positives are all substring "
      "matches: \"son\" inside person / season / reason, \"home\" inside "
      f"homework. The {fn} false negatives are critical information with "
      "none of the 13 keywords (pills, nurse, a phone number). "
      "\"grandson\" is a true positive only by the same substring accident.")
    w(f"- End-to-end recall is lower than (b) because {len(lost)} "
      "critical-level memories were deleted by the maintenance cycle's "
      "decay/cleanup *before* its alert hook ran (the ordering issue found "
      "in Phase 6), so they were never notified: "
      + "; ".join(f"\"{c}\"" for c in lost) + ".")
    w(f"- (c)'s higher precision ({pp:.2f}) is not an improvement: the "
      "same pre-alert pruning that drops real critical alerts also drops "
      "the critical-level false positives. Only warning-level alerts "
      "(0.3 ≤ R < 0.5) can reach a caregiver on the scheduled path.")
    w(f"- Scheduler failures during the run: {sched_fail}.\n")

    # 3 ────────────────────────────────────────────────────────
    w("## 3. Feedback-loop convergence (Phase 1)\n")
    w("**Method.** Signals sent through the real public API "
      "(`MemoryStore.record_feedback` / `record_task_outcome` → "
      "`MemoryOrchestrator` → `EpisodicMemoryStore`, persisted to SQLite). "
      f"Each update is an EMA step with α = {ALPHA} from the 0.5 default, "
      "so after n identical signals the value should be exactly "
      "1 − 0.5·0.7ⁿ (positive) or 0.5·0.7ⁿ (negative).\n")
    w("**(a) Repeated identical signals, 50 each:**\n")
    w("| Field | Signal | Max error vs closed form | Value after 50 | "
      "Signals to within 0.1 / 0.01 / 0.001 | Monotone | In [0,1] | "
      "Survives reload |")
    w("|---|---|---|---|---|---|---|---|")
    for (field, sig), s in fb["sequences"].items():
        st = s["steps"]
        w(f"| {field} | {'positive' if sig else 'negative'} | "
          f"{s['max_err']:.1e} | {s['final']:.2e} | "
          f"{st[0.1]} / {st[0.01]} / {st[0.001]} | "
          f"{'yes' if s['monotone'] else 'NO'} | "
          f"{'yes' if s['bounded'] else 'NO'} | "
          f"{'yes' if s['persisted'] else 'NO'} |")
    w("")
    w("**(b) Noisy signals.** 600 random signals drawn with probability p; "
      "statistics over the last 500 (first 100 discarded as burn-in). An EMA's "
      "long-run mean equals the rate of signals it received, with standard "
      "deviation √(α/(2−α)·p(1−p)). The reference is the rate actually "
      "sent: with 500 draws that rate itself wanders about ±0.02 around p, "
      "which is sampling noise, not EMA error.\n")
    w("| Field | p | Rate actually sent | EMA mean | EMA − sent | "
      "Observed SD | Theoretical SD |")
    w("|---|---|---|---|---|---|---|")
    for field, p_, obs, mean, sd, tsd in fb["noisy"]:
        w(f"| {field} | {p_} | {obs:.3f} | {mean:.3f} | {mean - obs:+.3f} | "
          f"{sd:.3f} | {tsd:.3f} |")
    w("")
    hi_v, lo_v, hi_t, lo_t = fb["alternating"]
    w(f"**(c) Alternating +/−, 200 signals:** settles into a fixed "
      f"2-cycle {hi_v:.4f} / {lo_v:.4f} (theory {hi_t:.4f} / {lo_t:.4f}). "
      "The value is bounded and does not drift.\n")
    base, up, down = fb["priority"]
    w(f"**(d) Effect on the DMS priority score** (identical memories, "
      f"importance 0.45, same age, private weights): baseline "
      f"{base:.3f}; after 20 positive + 20 successful signals "
      f"{up:.3f} ({(up / base - 1) * 100:+.0f}%); after 20 negative + 20 "
      f"failed signals {down:.3f} ({(down / base - 1) * 100:+.0f}%).\n")
    steps01 = fb["sequences"][("agent_feedback", True)]["steps"][0.01]
    worst = max(abs(m - o) for _, _, o, m, _, _ in fb["noisy"])
    sd_lo = min(sd for *_, sd, _ in fb["noisy"])
    sd_hi = max(sd for *_, sd, _ in fb["noisy"])
    w(f"**Interpretation.** One signal moves the value {ALPHA:.0%} of the "
      f"way to its target; {steps01} consistent signals bring it within "
      "0.01. Under noisy signals the long-run value tracks the rate of "
      f"signals received to within {worst:.3f}, while fluctuating around "
      f"it with SD {sd_lo:.2f}–{sd_hi:.2f} from turn to turn. Note that "
      "`PlannerAgent` records a *positive* signal every time it retrieves "
      "a memory that passed the relevance cut-off and was used in a turn "
      "whose LLM call succeeded (Phase 9; previously every top-k hit, even "
      "on failed turns), so `agent_feedback` now tracks how often a memory "
      "is genuinely useful. `task_success_rate` moves only through "
      "`POST /feedback`.\n")

    # 4 ────────────────────────────────────────────────────────
    if rq is not None:
        rows, sweep, best_cutoff, reembeds = rq
        w("## 4. Retrieval quality & efficiency\n")
        w("**Method.** `evaluation/retrieval_eval.py`: 20 user facts + 3 "
          "assistant replies for one user; 22 answerable questions with "
          "labelled gold memories and 8 no-answer queries (off-topic and "
          "small talk). Baseline = fixed top-5, no relevance cut-off, "
          "assistant replies retrievable, small talk retrieves. Phase 9 = "
          "intent-routed top-k, relevance cut-off, replies excluded, small "
          "talk skipped. Deterministic bag-of-words embedder (exact-word "
          "matching only, so recall on paraphrases is understated); run "
          "`python -m evaluation.retrieval_eval` for all-MiniLM-L6-v2 "
          "numbers.\n")
        w("| Configuration | Hit rate | Recall | Precision | F1 | MRR | "
          "Noise rejection | Irrelevant / query |")
        w("|---|---|---|---|---|---|---|---|")
        for name, m in rows:
            w(f"| {name} | {m['hit']:.1%} | {m['recall']:.1%} | "
              f"{m['precision']:.1%} | {m['f1']:.1%} | {m['mrr']:.3f} | "
              f"{m['noise_rejection']:.1%} | "
              f"{m['irrelevant_per_query']:.2f} |")
        w("")
        w(f"Stored memories re-embedded per query: **{reembeds:.1f}** "
          "(Phase 8: every stored memory, every query). Retrievals per "
          "chat turn: **1** (Phase 8: 2).\n")

    # gates ─────────────────────────────────────────────────────
    w("## Correctness gates\n")
    w("| Gate | Result | Detail |")
    w("|---|---|---|")
    for name, ok, detail in GATES:
        w(f"| {name} | {'pass' if ok else '**FAIL**'} | {detail} |")
    w("")
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def eval_retrieval():
    """Section 4 — retrieval benchmark + gates (runs last: it repoints
    the episodic tier's embedder at the deterministic test embedder)."""
    from evaluation import retrieval_eval
    from memory.retrieval_config import DEFAULT_MIN_SCORE

    rows, sweep, best_cutoff, reembeds = retrieval_eval.run_benchmark(
        embed, DEFAULT_MIN_SCORE)
    base, new = rows[0][1], rows[1][1]

    gate("Retrieval: no-answer queries inject nothing (Phase 9)",
         new["noise_rejection"] == 1.0,
         f"noise rejection {new['noise_rejection']:.0%} "
         f"(baseline {base['noise_rejection']:.0%})")
    gate("Retrieval: irrelevant memories per query cut by ≥ 75%",
         new["irrelevant_per_query"] <= 0.25 * base["irrelevant_per_query"],
         f"{base['irrelevant_per_query']:.2f} → "
         f"{new['irrelevant_per_query']:.2f}")
    gate("Retrieval: precision and F1 above baseline",
         new["precision"] > base["precision"] and new["f1"] > base["f1"],
         f"precision {base['precision']:.1%} → {new['precision']:.1%}, "
         f"F1 {base['f1']:.1%} → {new['f1']:.1%}")
    gate("Retrieval: assistant replies never served as memories",
         new.get("assistant_leaks", 1) == 0,
         f"{new.get('assistant_leaks')} leaks on replies-targeted queries")
    gate("Retrieval: no stored memory re-embedded at query time",
         reembeds == 0,
         f"{reembeds:.1f} re-embeddings per query")
    return rows, sweep, best_cutoff, reembeds


def main():
    ret = eval_retention()
    alr = eval_alerts()
    fb = eval_feedback()
    rq = eval_retrieval()
    write_report(ret, alr, fb, rq)
    failed = [g for g in GATES if not g[1]]
    print(f"Wrote {RESULTS_PATH}")
    print(f"Gates: {len(GATES) - len(failed)}/{len(GATES)} passed")
    for name, _, detail in failed:
        print(f"  FAIL: {name} ({detail})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
