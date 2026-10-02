# api/app_routes.py

"""
Application API (/app/...) used by the React frontend.

Two layers of access:
1. X-API-Key (MAEM_API_KEY) — unlocks the application; required on
   every /app route (enforced where this router is included).
2. Login (Bearer JWT) — identifies the person. A user only ever sees
   their own data; a caregiver only sees their linked Alzheimer's
   patients. Standard users can never be linked to a caregiver.
"""

import math
import os
import re
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from accounts.security import TokenError, create_token, decode_token
from accounts.store import (
    ROLE_CAREGIVER,
    TYPE_ALZHEIMERS,
    AccountError,
    AccountStore,
)
from memory.notifier import CRITICAL_KEYWORDS, detect_alerts

# Memory health bands (retention)
STRONG, FADING = 0.7, 0.5

TODAY_WORDS = [
    "today", "tomorrow", "tonight", "morning", "evening", "every day",
    "daily", "appointment", "doctor", "medicine", "medication", "tablet",
    "insulin", "visit", "visits", "meeting", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
]

# Baseline stability (hours) of an ordinary memory: 24 h × (1 + 2 × 0.45)
BASE_STABILITY = 45.6


# ─────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    username: str
    password: str
    account_type: str               # standard | alzheimers | caregiver
    severity: Optional[str] = None  # mild | moderate | severe
    caregiver: Optional[str] = None


class PasswordRequest(BaseModel):
    current_password: str
    new_password: str


class AppChatRequest(BaseModel):
    message: str


class AddMemoryRequest(BaseModel):
    text: str
    important: bool = False
    user: Optional[str] = None


class ReminderRequest(BaseModel):
    title: str
    due_at: Optional[str] = None
    user: Optional[str] = None


class ReminderUpdate(BaseModel):
    done: bool


class NotificationSettings(BaseModel):
    email_enabled: bool
    email: Optional[str] = None


class SimulateRequest(BaseModel):
    hours: float = 12.0
    user: Optional[str] = None


class ResetRequest(BaseModel):
    user: Optional[str] = None


class ResetAllRequest(BaseModel):
    api_key: str


# ─────────────────────────────────────────────────────────────
# Router factory
# ─────────────────────────────────────────────────────────────

def build_app_router(orch, accounts: AccountStore, limiter) -> APIRouter:

    router = APIRouter(prefix="/app", tags=["application"])
    bearer = HTTPBearer(auto_error=False)

    def memory():
        return orch.memory_store.orchestrator

    # ── auth helpers ─────────────────────────────────────────

    def current_user(
        creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    ) -> Dict[str, Any]:
        if creds is None or creds.scheme.lower() != "bearer":
            raise HTTPException(401, "Please log in.")
        try:
            payload = decode_token(creds.credentials)
        except TokenError as exc:
            raise HTTPException(401, str(exc)) from exc
        user = accounts.get_user(payload.get("sub", ""))
        if user is None:
            raise HTTPException(401, "Account no longer exists.")
        return user

    def caregiver_only(user=Depends(current_user)):
        if user["role"] != ROLE_CAREGIVER:
            raise HTTPException(403, "Caregiver accounts only.")
        return user

    def target_of(viewer: Dict[str, Any], requested: Optional[str]) -> str:
        """Whose data this request is about, after access checks."""
        if not requested:
            if viewer["role"] == ROLE_CAREGIVER:
                raise HTTPException(400, "Choose a patient.")
            return viewer["username"]
        requested = requested.strip().lower()
        if not accounts.can_access(viewer, requested):
            raise HTTPException(403, "You don't have access to that account.")
        return requested

    def bad_request(exc: Exception):
        raise HTTPException(400, str(exc)) from exc

    # ── memory helpers ───────────────────────────────────────

    def episodes_of(username: str):
        with memory().episodic_lock:
            eps = memory().episodic_memory.get_all()
        out = []
        for ep in eps:
            ctx = ep.context if isinstance(ep.context, dict) else {}
            if ep.agent_id == username and ctx.get("role") != "assistant":
                out.append(ep)
        return out

    def health(retention: float) -> str:
        if retention >= STRONG:
            return "strong"
        if retention >= FADING:
            return "fading"
        return "at_risk"

    def memory_view(ep) -> Dict[str, Any]:
        r = ep.retention()
        ctx = ep.context if isinstance(ep.context, dict) else {}
        return {
            "id": ep.episode_id,
            "content": str(ep.content),
            "retention": round(r, 3),
            "importance": round(ep.importance, 2),
            "priority": round(ep.priority_score(), 3),
            "times_retrieved": ep.review_count,
            "stability_hours": round(ep.stability_hours, 1),
            "created_at": ep.created_at.isoformat(),
            "last_recalled": ep.last_reviewed_at.isoformat(),
            "health": health(r),
            "important": bool(ctx.get("important")),
            "source": ctx.get("source", "chat"),
        }

    def embed(text: str):
        fn = getattr(orch, "_embed", None) or orch.executor.embed_fn
        return fn(text)

    def summary(username: str) -> Dict[str, Any]:
        eps = episodes_of(username)
        rs = [ep.retention() for ep in eps]
        return {
            "memory_count": len(eps),
            "avg_retention": round(sum(rs) / len(rs), 3) if rs else None,
            "strong": sum(1 for r in rs if r >= STRONG),
            "fading": sum(1 for r in rs if FADING <= r < STRONG),
            "at_risk": sum(1 for r in rs if r < FADING),
        }

    def refresh_alerts(username: str) -> None:
        with memory().episodic_lock:
            eps = memory().episodic_memory.get_all()
        accounts.sync_alerts(username, detect_alerts(eps, username))

    def wipe_memories(usernames: List[str]) -> int:
        mo = memory()
        removed = 0
        targets = set(usernames)
        with mo.episodic_lock:
            doomed = [ep.episode_id for ep in mo.episodic_memory.get_all()
                      if ep.agent_id in targets]
            for eid in doomed:
                mo.episodic_memory.delete(eid)
        with mo.semantic_lock:
            for eid in doomed:
                try:
                    mo.semantic_memory.delete(eid)
                except Exception:
                    pass
        with mo.working_lock:
            for u in targets:
                mo.working_memory.clear(u)
        removed += len(doomed)
        notified = getattr(getattr(orch, "notifier", None),
                           "_already_notified", None)
        if isinstance(notified, set):
            for key in [k for k in notified if k[0] in targets]:
                notified.discard(key)
        return removed

    def public_user(user: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(user)
        out.pop("created_at", None)
        if user["role"] == ROLE_CAREGIVER:
            out["patients"] = [
                {"username": p["username"], "name": p["name"],
                 "severity": p["severity"]}
                for p in accounts.patients_of(user["username"])]
        elif user["account_type"] == TYPE_ALZHEIMERS:
            cg = accounts.get_user(accounts.caregiver_of(user["username"]) or "")
            out["caregiver"] = (
                {"username": cg["username"], "name": cg["name"]} if cg else None)
        return out

    # ═════════════════════════════════════════════════════════
    # Auth
    # ═════════════════════════════════════════════════════════

    @router.post("/auth/login")
    @limiter.limit("10/minute")
    def login(request: Request, req: LoginRequest):
        user = accounts.authenticate(req.username, req.password)
        if user is None:
            raise HTTPException(401, "Wrong username or password.")
        return {"token": create_token(user["username"], user["role"]),
                "user": public_user(user)}

    @router.post("/auth/register")
    @limiter.limit("10/minute")
    def register(request: Request, req: RegisterRequest):
        try:
            user = accounts.create_user(
                req.name, req.username, req.password, req.account_type,
                severity=req.severity, caregiver=req.caregiver)
        except AccountError as exc:
            bad_request(exc)
        if user["role"] != ROLE_CAREGIVER:
            memory().episodic_memory.set_agent_decay_multiplier(
                user["username"], user["decay_multiplier"])
        return {"token": create_token(user["username"], user["role"]),
                "user": public_user(user)}

    @router.get("/auth/caregivers")
    def caregivers():
        """For the registration form (Alzheimer's accounts pick one)."""
        return [{"username": u["username"], "name": u["name"]}
                for u in accounts.list_users() if u["role"] == ROLE_CAREGIVER]

    @router.get("/auth/me")
    def me(user=Depends(current_user)):
        return public_user(user)

    @router.post("/auth/password")
    def change_password(req: PasswordRequest, user=Depends(current_user)):
        try:
            accounts.change_password(user["username"], req.current_password,
                                     req.new_password)
        except AccountError as exc:
            bad_request(exc)
        return {"status": "ok"}

    # ═════════════════════════════════════════════════════════
    # Chat
    # ═════════════════════════════════════════════════════════

    @router.post("/chat")
    @limiter.limit("30/minute")
    def chat(request: Request, req: AppChatRequest, user=Depends(current_user)):
        if user["role"] == ROLE_CAREGIVER:
            raise HTTPException(403, "Chat is for user accounts.")
        try:
            result = orch.route(task=req.message, agent_id=user["username"])
        except ValueError as exc:
            bad_request(exc)
        retrieved = result.get("retrieved_memories", {}) or {}
        scores = retrieved.get("scores") or {}
        used = []
        for m in retrieved.get("fused_memory", []):
            mid = getattr(m, "episode_id", None) or getattr(m, "memory_id", None)
            used.append({"content": str(m.content),
                         "relevance": scores.get(mid)})
        return {"reply": result["response"],
                "intent": result.get("intent"),
                "memories_used": used}

    # ═════════════════════════════════════════════════════════
    # Memories
    # ═════════════════════════════════════════════════════════

    @router.get("/memories")
    def list_memories(user: Optional[str] = None, viewer=Depends(current_user)):
        target = target_of(viewer, user)
        items = [memory_view(ep) for ep in episodes_of(target)]
        items.sort(key=lambda m: m["created_at"], reverse=True)
        return {"user": target, "summary": summary(target), "memories": items}

    @router.post("/memories")
    def add_memory(req: AddMemoryRequest, viewer=Depends(current_user)):
        target = target_of(viewer, req.user)
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(400, "Memory text is required.")
        source = "caregiver" if viewer["role"] == ROLE_CAREGIVER else "manual"
        try:
            stored = orch.memory_store.store(
                agent_id=target, content=text, embedding=embed(text),
                context={"role": "user", "source": source,
                         "important": bool(req.important)})
        except ValueError as exc:
            bad_request(exc)
        mo = memory()
        with mo.episodic_lock:
            ep = mo.episodic_memory.get_by_id(stored["memory_id"])
            if ep is not None and req.important:
                ep.importance = max(ep.importance, 0.9)
                ep.stability_hours = max(ep.stability_hours, 24.0 * (1 + 2 * 0.9))
                mo.episodic_memory.save(ep)
        return memory_view(ep) if ep is not None else {"id": stored["memory_id"]}

    @router.post("/memories/{memory_id}/reinforce")
    def reinforce(memory_id: str, viewer=Depends(current_user)):
        ep = memory().episodic_memory.get_by_id(memory_id)
        if ep is None:
            raise HTTPException(404, "Memory not found.")
        target_of(viewer, ep.agent_id)
        orch.memory_store.reinforce(agent_id=ep.agent_id, memory_id=memory_id)
        return memory_view(memory().episodic_memory.get_by_id(memory_id))

    # ═════════════════════════════════════════════════════════
    # Reminders
    # ═════════════════════════════════════════════════════════

    @router.get("/reminders")
    def list_reminders(user: Optional[str] = None, viewer=Depends(current_user)):
        return accounts.list_reminders(target_of(viewer, user))

    @router.post("/reminders")
    def add_reminder(req: ReminderRequest, viewer=Depends(current_user)):
        target = target_of(viewer, req.user)
        try:
            return accounts.add_reminder(target, req.title, req.due_at,
                                         viewer["username"])
        except AccountError as exc:
            bad_request(exc)

    def _own_reminder(reminder_id: str, viewer) -> Dict[str, Any]:
        rem = accounts.get_reminder(reminder_id)
        if rem is None:
            raise HTTPException(404, "Reminder not found.")
        target_of(viewer, rem["username"])
        return rem

    @router.patch("/reminders/{reminder_id}")
    def update_reminder(reminder_id: str, req: ReminderUpdate,
                        viewer=Depends(current_user)):
        _own_reminder(reminder_id, viewer)
        accounts.set_reminder_done(reminder_id, req.done)
        return accounts.get_reminder(reminder_id)

    @router.delete("/reminders/{reminder_id}")
    def delete_reminder(reminder_id: str, viewer=Depends(current_user)):
        _own_reminder(reminder_id, viewer)
        accounts.delete_reminder(reminder_id)
        return {"status": "deleted"}

    # ═════════════════════════════════════════════════════════
    # Today (card + gentle reminders)
    # ═════════════════════════════════════════════════════════

    @router.get("/today")
    def today(user: Optional[str] = None, viewer=Depends(current_user)):
        target = target_of(viewer, user)
        today_iso = datetime.now().date().isoformat()

        reminders = [
            r for r in accounts.list_reminders(target)
            if not r["done"] and (
                r["due_at"] is None or r["due_at"][:10] <= today_iso)
        ]

        eps = [ep for ep in episodes_of(target) if not ep.is_forgotten()]
        relevant = []
        for ep in eps:
            text = str(ep.content).lower()
            if any(re.search(rf"\b{re.escape(w)}\b", text) for w in TODAY_WORDS):
                relevant.append(ep)
        relevant.sort(key=lambda ep: ep.priority_score(), reverse=True)

        gentle = []
        owner = accounts.get_user(target)
        if owner and owner["account_type"] == TYPE_ALZHEIMERS:
            fading = [
                ep for ep in eps
                if ep.retention() < 0.6 and any(
                    kw in str(ep.content).lower() for kw in CRITICAL_KEYWORDS)
            ]
            fading.sort(key=lambda ep: ep.retention())
            gentle = [{"id": ep.episode_id,
                       "text": f"You told me: “{ep.content}”"}
                      for ep in fading[:3]]

        return {
            "date": today_iso,
            "reminders": reminders,
            "memories": [memory_view(ep) for ep in relevant[:5]],
            "gentle_reminders": gentle,
        }

    # ═════════════════════════════════════════════════════════
    # Caregiver
    # ═════════════════════════════════════════════════════════

    @router.get("/patients")
    def patients(viewer=Depends(caregiver_only)):
        out = []
        for p in accounts.patients_of(viewer["username"]):
            refresh_alerts(p["username"])
            out.append({
                "username": p["username"], "name": p["name"],
                "severity": p["severity"],
                "decay_multiplier": p["decay_multiplier"],
                "open_alerts": accounts.open_alert_count(p["username"]),
                **summary(p["username"]),
            })
        return out

    @router.get("/alerts")
    def alerts(user: str, status: Optional[str] = None,
               viewer=Depends(caregiver_only)):
        target = target_of(viewer, user)
        refresh_alerts(target)
        return accounts.list_alerts(target, status)

    @router.post("/alerts/{alert_id}/handle")
    def handle_alert(alert_id: str, viewer=Depends(caregiver_only)):
        alert = accounts.get_alert(alert_id)
        if alert is None:
            raise HTTPException(404, "Alert not found.")
        target_of(viewer, alert["username"])
        accounts.handle_alert(alert_id, viewer["username"])
        return accounts.get_alert(alert_id)

    @router.get("/trend")
    def trend(user: Optional[str] = None, days: int = 14,
              viewer=Depends(current_user)):
        target = target_of(viewer, user)
        s = summary(target)
        if s["memory_count"]:
            accounts.record_snapshot(target, s["avg_retention"],
                                     s["memory_count"], s["at_risk"])
        return accounts.snapshots(target, max(1, min(days, 90)))

    @router.get("/settings/notifications")
    def get_notifications(viewer=Depends(caregiver_only)):
        out = accounts.get_notifications(viewer["username"])
        out["smtp_configured"] = bool(getattr(orch.notifier, "smtp_ready", False)) \
            if hasattr(orch, "notifier") else False
        return out

    @router.put("/settings/notifications")
    def set_notifications(req: NotificationSettings,
                          viewer=Depends(caregiver_only)):
        try:
            return accounts.set_notifications(
                viewer["username"], req.email_enabled, req.email)
        except AccountError as exc:
            bad_request(exc)

    # ═════════════════════════════════════════════════════════
    # Memory tools (Settings)
    # ═════════════════════════════════════════════════════════

    @router.post("/tools/simulate")
    def simulate(req: SimulateRequest, viewer=Depends(current_user)):
        target = target_of(viewer, req.user)
        if not (0 < req.hours <= 720):
            raise HTTPException(400, "Hours must be between 1 and 720.")
        mo = memory()
        with mo.episodic_lock:
            aged = mo.episodic_memory.simulate_time_passage(target, req.hours)
        check = getattr(orch, "check_caregiver_alerts", None)
        if callable(check):
            check()
        return {"user": target, "hours": req.hours, "memories_aged": aged,
                "summary": summary(target)}

    @router.get("/tools/compare")
    def compare(viewer=Depends(current_user)):
        """
        Forgetting curves for normal memory vs each Alzheimer's
        severity, plus the accounts this viewer may see (their own, or
        a caregiver's patients) — aggregates only, never contents.
        """
        def curve(multiplier):
            s = BASE_STABILITY / multiplier
            return [{"hours": h, "retention": round(math.exp(-h / s), 4)}
                    for h in range(0, 121, 6)]

        curves = [
            {"label": "Standard memory", "multiplier": 1.0, "points": curve(1.0)},
            {"label": "Mild", "multiplier": 1.5, "points": curve(1.5)},
            {"label": "Moderate", "multiplier": 3.0, "points": curve(3.0)},
            {"label": "Severe", "multiplier": 5.0, "points": curve(5.0)},
        ]
        if viewer["role"] == ROLE_CAREGIVER:
            visible = accounts.patients_of(viewer["username"])
        else:
            visible = [viewer]
        people = [{
            "username": u["username"], "name": u["name"],
            "account_type": u["account_type"], "severity": u["severity"],
            "decay_multiplier": u["decay_multiplier"],
            **summary(u["username"]),
        } for u in visible]
        return {"curves": curves, "accounts": people}

    # ═════════════════════════════════════════════════════════
    # Reset
    # ═════════════════════════════════════════════════════════

    @router.post("/reset")
    def reset(req: ResetRequest, viewer=Depends(current_user)):
        target = target_of(viewer, req.user)
        removed = wipe_memories([target])
        accounts.delete_user_data(target)
        return {"status": "reset", "user": target, "memories_removed": removed}

    @router.post("/reset-all")
    def reset_all(req: ResetAllRequest, viewer=Depends(current_user)):
        expected = os.getenv("MAEM_API_KEY", "")
        if not expected or not secrets.compare_digest(req.api_key, expected):
            raise HTTPException(403, "Re-enter the correct MAEM API key.")
        with memory().episodic_lock:
            agents = {ep.agent_id for ep in memory().episodic_memory.get_all()}
        agents |= {u["username"] for u in accounts.list_users()}
        removed = wipe_memories(sorted(agents))
        accounts.delete_all_data()
        return {"status": "reset", "memories_removed": removed}

    return router
