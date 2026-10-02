"""
FastAPI entry point for the
multi-agent memory system.
"""
import logging
import secrets
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import FileResponse
import os
import sys

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from orchestrator.orchestrator import Orchestrator
from memory.notifier import detect_alerts
from accounts.store import AccountStore
from api.app_routes import build_app_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MultiAgent Memory System",
    version="1.0.0"
)

# ── Rate limiting ──────────────────────────────────────────────
# Keyed by client IP. Applied selectively (see /chat and /store
# below) rather than globally, since some routes (health check,
# scheduler status) are cheap and safe to poll more often.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ─────────────────────────────────────────────────────
# Origins come from ALLOWED_ORIGINS (comma-separated) so this can
# differ per environment without a code change. Defaults to
# localhost-only so an unconfigured deployment isn't wide open.
_DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", _DEFAULT_ALLOWED_ORIGINS
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API key auth ─────────────────────────────────────────────
# A single shared-secret header, checked against MAEM_API_KEY.
# Applied to every route except the GET / health check (see the
# `router` below, which every other route is registered on).
API_KEY_HEADER_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def verify_api_key(
    api_key: Optional[str] = Security(_api_key_header)
) -> str:
    expected = os.getenv("MAEM_API_KEY")

    if not expected:
        # Fail closed: an unconfigured server should not silently
        # accept every request.
        logger.error(
            "MAEM_API_KEY is not set — rejecting request; "
            "set MAEM_API_KEY in the environment to enable access."
        )
        raise HTTPException(
            status_code=500,
            detail="Server is not configured with an API key."
        )

    if not api_key or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key."
        )

    return api_key


# All routes except the health check live on this router, so the
# API-key dependency can't be forgotten on a newly added route.
router = APIRouter(dependencies=[Depends(verify_api_key)])

# ── Single orchestrator — owns the memory stack ──
orch = Orchestrator()

# ── Application accounts (users, caregivers, reminders, alerts) ──
accounts = getattr(orch, "accounts", None) or AccountStore()


@app.on_event("startup")
def seed_accounts():
    """Create the default accounts on first run and apply decay speeds."""
    if accounts.ensure_seeded():
        logger.info("Seeded default application accounts.")
    apply = getattr(orch, "apply_account_decay_multipliers", None)
    if callable(apply):
        apply()
    else:
        episodic = orch.memory_store.orchestrator.episodic_memory
        for username, m in accounts.decay_multipliers().items():
            episodic.set_agent_decay_multiplier(username, m)


# ── Request models ──
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"


class StoreRequest(BaseModel):
    agent_id: str = "default"
    content: str
    embedding: List[float]
    context: Optional[Dict[str, Any]] = None


class RetrieveRequest(BaseModel):
    agent_id: str = "default"
    query: str
    embedding: List[float]
    top_k: int = 5


class ReinforceRequest(BaseModel):
    agent_id: str = "default"
    memory_id: str


class FeedbackRequest(BaseModel):
    positive: Optional[bool] = None
    success: Optional[bool] = None


# ── Routes ──

@app.get("/")
def root():
    """Unauthenticated health check — intentionally not on `router`."""
    return {
        "status": "running",
        "system": "MultiAgent Memory System"
    }


@router.post("/chat")
@limiter.limit("30/minute")
def chat(request: Request, req: ChatRequest):
    """
    Main chat route — routes through
    PlannerAgent → ExecutorAgent.
    """
    # Input validation (empty message) raises ValueError deep inside the
    # planner; surface it as a client error, not a bare 500. Anything
    # else (e.g. the LLM call failing) still propagates as a 500.
    try:
        result = orch.route(task=req.message, agent_id=req.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    fused = result.get(
        "retrieved_memories", {}
    ).get("fused_memory", [])
    return {
        "reply": result["response"],
        "agent_used": "executor",
        "memories_retrieved": len(fused),
        "routing_path": ["planner", "executor"]
    }


# ── KEY ROUTE — live memory panel ──
@router.get("/metrics/{agent_id}")
def get_metrics(agent_id: str):
    """
    Live retention, stability, priority
    scores for the memory panel.
    """
    try:
        result = (
            orch.memory_store
            .orchestrator
            .metrics(agent_id=agent_id)
        )
        return result
    except Exception as e:
        logger.error("Metrics error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/memories/{agent_id}")
def get_memories(agent_id: str):
    try:
        result = (
            orch.memory_store
            .orchestrator
            .metrics(agent_id=agent_id)
        )
        return {"episodes": result["memories"]}
    except Exception as e:
        logger.error("Memories error: %s", e)
        return {"episodes": []}


@router.post("/store")
@limiter.limit("30/minute")
def store_memory(request: Request, req: StoreRequest):
    try:
        memory_ids = orch.memory_store.store(
            agent_id=req.agent_id,
            content=req.content,
            embedding=req.embedding,
            context=req.context
        )
    except ValueError as e:
        # Bad client input (empty content/agent_id, wrong embedding
        # dimension, NaN, invalid role/shared). Validation runs before
        # any tier is written, so nothing was stored.
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "status": "stored",
        "memory_ids": memory_ids
    }


@router.post("/retrieve")
def retrieve_memories(req: RetrieveRequest):
    try:
        results = orch.memory_store.retrieve(
            agent_id=req.agent_id,
            query=req.query,
            embedding=req.embedding,
            top_k=req.top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return results


@router.get("/context/{agent_id}")
def get_context(agent_id: str, limit: int = 10):
    return orch.memory_store.get_context(
        agent_id=agent_id,
        limit=limit
    )


@router.post("/reinforce")
def reinforce_memory(req: ReinforceRequest):
    orch.memory_store.reinforce(
        agent_id=req.agent_id,
        memory_id=req.memory_id
    )
    return {
        "status": "reinforced",
        "memory_id": req.memory_id
    }


@router.post("/feedback/{agent_id}/{memory_id}")
def submit_feedback(
    agent_id: str,
    memory_id: str,
    req: FeedbackRequest
):
    """
    Record real feedback/outcome signal for a memory, updating
    Episode.agent_feedback and/or Episode.task_success_rate (an
    EMA nudge toward 1.0 on positive/success, toward 0.0 on
    negative/failure). At least one of "positive"/"success" must
    be provided.
    """
    if req.positive is None and req.success is None:
        raise HTTPException(
            status_code=400,
            detail="provide at least one of 'positive' or 'success'"
        )

    feedback_updated = None
    outcome_updated = None

    if req.positive is not None:
        feedback_updated = orch.memory_store.record_feedback(
            agent_id=agent_id,
            memory_id=memory_id,
            positive=req.positive
        )

    if req.success is not None:
        outcome_updated = orch.memory_store.record_task_outcome(
            agent_id=agent_id,
            memory_id=memory_id,
            success=req.success
        )

    attempted_updates = [
        v for v in (feedback_updated, outcome_updated) if v is not None
    ]
    if attempted_updates and not any(attempted_updates):
        raise HTTPException(
            status_code=404,
            detail=f"no memory found with id {memory_id}"
        )

    return {
        "status": "ok",
        "agent_id": agent_id,
        "memory_id": memory_id,
        "feedback_updated": feedback_updated,
        "outcome_updated": outcome_updated
    }


@router.post("/decay")
def trigger_decay():
    orch.memory_store.apply_decay()
    return {"status": "decay applied"}


@router.post("/cleanup")
def cleanup_memories():
    removed = orch.memory_store.cleanup()
    return {
        "status": "cleanup complete",
        "removed": removed
    }


@router.get("/scheduler/status")
def scheduler_status():
    return {
        "running": orch.scheduler.is_running()
    }


_API_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIST = os.path.join(
    os.path.dirname(_API_DIR), "frontend", "dist"
)
_LEGACY_UI = os.path.join(_API_DIR, "maem_multiagent.html")


def _ui_index():
    built = os.path.join(_FRONTEND_DIST, "index.html")
    return built if os.path.isfile(built) else _LEGACY_UI


@app.get("/ui")
@app.get("/ui/", include_in_schema=False)
def serve_ui():
    """
    Serves the web application. Deliberately NOT behind
    verify_api_key: a plain browser navigation can't attach a custom
    X-API-Key header. The page carries no secrets or memory data —
    every data call it makes still goes through the API-key (and, for
    /app routes, login) checks.

    Serves the React build (frontend/dist) when it exists, otherwise
    the legacy single-file dashboard.
    """
    return FileResponse(_ui_index())


@app.get("/ui/{asset_path:path}", include_in_schema=False)
def serve_ui_asset(asset_path: str):
    """Static files of the React build; unknown paths fall back to the app."""
    root = os.path.realpath(_FRONTEND_DIST)
    candidate = os.path.realpath(os.path.join(root, asset_path))
    if candidate.startswith(root + os.sep) and os.path.isfile(candidate):
        return FileResponse(candidate)
    return FileResponse(_ui_index())


@app.on_event("shutdown")
def shutdown():
    orch.stop()
    logger.info("Scheduler stopped.")

class SimulateTimeRequest(BaseModel):
    agent_id: str
    hours: float = 12.0

@router.post("/simulate_time")
def simulate_time(req: SimulateTimeRequest):
    try:
        memory = orch.memory_store.orchestrator
        with memory.episodic_lock:
            aged = memory.episodic_memory.simulate_time_passage(
                agent_id=req.agent_id,
                hours=req.hours
            )
        logger.debug(
            "simulate_time: agent=%s hours=%s aged=%d",
            req.agent_id, req.hours, aged
        )

        # Immediately check for caregiver alerts after aging, instead
        # of waiting for the scheduler's next 5-minute cycle — so
        # newly-critical memories get notified right away.
        orch.check_caregiver_alerts()

        return {"status": "ok", "hours_aged": req.hours}
    except Exception as e:
        logger.exception("simulate_time error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/alerts/{agent_id}")
def get_alerts(agent_id: str):
    """
    Get caregiver alerts for agent.
    """
    try:
        with orch.memory_store.orchestrator.episodic_lock:
            episodes = orch.memory_store.orchestrator.episodic_memory.get_all()

        alerts = detect_alerts(episodes, agent_id)

        return {
            "agent_id": agent_id,
            "alert_count": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        logger.error("Alerts error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{agent_id}/notify")
def notify_alerts(agent_id: str):
    """
    Check the same critical/warning memories as GET /alerts,
    and send a caregiver notification for any that are new
    (email if configured via ENABLE_EMAIL_ALERTS, otherwise
    just logged). Useful for demoing/testing the notification
    path on demand, independent of the 5-minute scheduler cycle.
    """
    try:
        with orch.memory_store.orchestrator.episodic_lock:
            episodes = orch.memory_store.orchestrator.episodic_memory.get_all()

        alerts = detect_alerts(episodes, agent_id)
        sent = orch.notifier.notify(agent_id, alerts)

        return {
            "agent_id": agent_id,
            "alert_count": len(alerts),
            "notifications_sent": sent,
            "email_enabled": orch.notifier.email_enabled,
            "sms_enabled": orch.notifier.sms_enabled
        }
    except Exception as e:
        logger.error("Notify error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class EvalRequest(BaseModel):
    agent_id: str = "alice"
    test_questions: List[str] = []
    # Optional, one per question: what a correct answer must contain.
    # Alternatives separated by "|", e.g. "Ravi|Ravi Kumar". When given,
    # answers are scored for correctness, not just memory use.
    expected_answers: Optional[List[str]] = None


_BASELINE_FAILED_KEYWORDS = [
    "don't have", "no information",
    "don't know", "cannot", "no context",
    "no memory", "no previous", "no stored",
    "haven't", "not aware", "no access"
]


def _answer_correct(response: str, expected: Optional[str]) -> Optional[bool]:
    if not expected or not expected.strip():
        return None
    text = response.lower()
    return any(
        alt.strip().lower() in text
        for alt in expected.split("|")
        if alt.strip()
    )


def _rate(values: List[Optional[bool]]) -> Optional[float]:
    scored = [v for v in values if v is not None]
    if not scored:
        return None
    return round(sum(1 for v in scored if v) / len(scored), 3)


@router.post("/evaluate")
def evaluate(req: EvalRequest):
    """
    Memory vs no-memory comparison on the given questions.

    For each question, MAEM answers through the full pipeline (intent
    routing, relevance-filtered retrieval) and a baseline answers the
    same question with no memory context. Reports:

    - maem_recall_rate      share of questions where relevant memories
                            were retrieved and used
    - baseline_recall_rate  share where the no-memory baseline still
                            claimed to know (keyword heuristic)
    - maem_accuracy /       share of answers containing the expected
      baseline_accuracy     answer (only when expected_answers given)
    - mean_relevance        average cosine relevance of the memories
                            that were injected
    """
    try:
        expected = list(req.expected_answers or [])
        if expected and len(expected) != len(req.test_questions):
            raise HTTPException(
                status_code=400,
                detail="expected_answers must have one entry per question"
            )

        results = []
        relevances: List[float] = []
        for i, question in enumerate(req.test_questions):
            # MAEM response (with memory)
            maem_result = orch.route(
                task=question,
                agent_id=req.agent_id
            )
            maem_response = maem_result["response"]
            retrieved = maem_result.get("retrieved_memories", {}) or {}
            fused = retrieved.get("fused_memory", [])
            memories_used = len(fused)
            scores = list((retrieved.get("scores") or {}).values())
            relevances.extend(scores)

            # Baseline response (no memory — empty context)
            baseline_response = orch.planner.executor.llm_fn(
                question,
                []  # ← empty context, no memories
            )

            baseline_recalled = not any(
                kw in baseline_response.lower()
                for kw in _BASELINE_FAILED_KEYWORDS
            )
            expected_i = expected[i] if expected else None

            results.append({
                "question": question,
                "intent": maem_result.get("intent"),
                "maem_response": maem_response[:200],
                "baseline_response": baseline_response[:200],
                "memories_used": memories_used,
                "memory_relevance": (
                    round(sum(scores) / len(scores), 3) if scores else None
                ),
                "maem_used_memory": memories_used > 0,
                "baseline_recalled": baseline_recalled,
                "expected": expected_i,
                "maem_correct": _answer_correct(maem_response, expected_i),
                "baseline_correct": _answer_correct(
                    baseline_response, expected_i
                ),
            })

        total = len(results)
        maem_recall = sum(1 for r in results if r["maem_used_memory"])
        baseline_recall = sum(1 for r in results if r["baseline_recalled"])

        return {
            "total_questions": total,
            "maem_recall_rate": round(maem_recall / total, 3) if total > 0 else 0,
            "baseline_recall_rate": round(baseline_recall / total, 3) if total > 0 else 0,
            "improvement": round((maem_recall - baseline_recall) / total, 3) if total > 0 else 0,
            "maem_accuracy": _rate([r["maem_correct"] for r in results]),
            "baseline_accuracy": _rate([r["baseline_correct"] for r in results]),
            "mean_relevance": (
                round(sum(relevances) / len(relevances), 3)
                if relevances else None
            ),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Eval error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Every route defined on `router` above requires X-API-Key; only
# GET / (defined directly on `app`) does not.
app.include_router(router)

# Application API for the web app: needs the API key (app unlock) AND a
# login token (see api/app_routes.py).
app.include_router(
    build_app_router(orch, accounts, limiter),
    dependencies=[Depends(verify_api_key)],
)
