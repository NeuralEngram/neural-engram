# tests/test_integration_alerts.py
"""
Caregiver alert pipeline, end-to-end:

  route() stores a critical-keyword memory
    → memory aged via last_reviewed_at (exactly what /simulate_time does)
    → Orchestrator._check_caregiver_alerts()
    → detect_alerts()
    → CaregiverNotifier.notify()
    → email sent (fake SMTP server) + recorded in _already_notified
    → second check does NOT re-send

Real Orchestrator, real notifier, real scheduler (driven with
run_once()). Only smtplib.SMTP is replaced, by a recorder.
"""

import importlib
import sys
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import memory.notifier as notifier_module
from memory.notifier import detect_alerts
from tests.conftest import TEST_API_KEY


CRITICAL = "my daughter Priya brings my medicine every morning"
HARMLESS = "I enjoy painting landscapes by the lake"

EMAIL_ENV = {
    "ENABLE_EMAIL_ALERTS": "true",
    "SMTP_HOST": "smtp.test.invalid",
    "SMTP_USER": "maem@test.invalid",
    "SMTP_PASSWORD": "not-a-password",
    "ALERT_EMAIL_TO": "caregiver@test.invalid",
}


class FakeSMTP:
    """Records sendmail() calls instead of talking to a server."""

    sent = []
    fail = False

    def __init__(self, host, port, timeout=None):
        self.host = host

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        pass

    def sendmail(self, from_addr, to_addrs, message):
        if FakeSMTP.fail:
            raise OSError("SMTP down")
        FakeSMTP.sent.append(message)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.sent = []
    FakeSMTP.fail = False
    monkeypatch.setattr(notifier_module.smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


@pytest.fixture
def emailing_orch(real_orchestrator, smtp):
    orch = real_orchestrator(env=EMAIL_ENV)
    assert orch.notifier.email_enabled
    return orch


# ─── helpers ───────────────────────────────────────────────

def episode_with(orch, text, agent_id="rajan"):
    return next(
        ep for ep in orch.memory_store.orchestrator.episodic_memory.get_all()
        if ep.content == text and ep.agent_id == agent_id
    )


def hours_to_retention(ep, target):
    """
    Smallest whole number of extra hours of ageing (from now, on top of
    any ageing already applied) that puts R(t) below target.
    """
    now = datetime.now(timezone.utc)
    lo, hi = 0, 1
    while ep.retention(now + timedelta(hours=hi)) >= target:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ep.retention(now + timedelta(hours=mid)) >= target:
            lo = mid
        else:
            hi = mid
    return hi


def simulate_time(orch, agent_id, hours):
    """Same mutation POST /simulate_time performs (without its check)."""
    mo = orch.memory_store.orchestrator
    with mo.episodic_lock:
        for ep in mo.episodic_memory.get_all():
            if ep.agent_id == agent_id:
                ep.last_reviewed_at -= timedelta(hours=hours)
                mo.episodic_memory._save_episode(ep)


def age_to(orch, text, target, agent_id="rajan"):
    ep = episode_with(orch, text, agent_id)
    simulate_time(orch, agent_id, hours_to_retention(ep, target))
    return ep


def key(ep, level, agent_id="rajan"):
    return (agent_id, str(ep.content)[:80], level)


# ─── Core pipeline ─────────────────────────────────────────

class TestAlertPipeline:

    @pytest.mark.parametrize(
        "target, level", [(0.45, "warning"), (0.2, "critical")]
    )
    def test_aged_critical_memory_is_notified_once(
        self, emailing_orch, smtp, target, level
    ):
        orch = emailing_orch
        orch.route(CRITICAL, agent_id="rajan")
        ep = age_to(orch, CRITICAL, target)

        orch._check_caregiver_alerts(None)

        assert key(ep, level) in orch.notifier._already_notified
        assert len(smtp.sent) == 1
        assert "Priya" in smtp.sent[0]
        assert level.upper() in smtp.sent[0]

        before = set(orch.notifier._already_notified)
        orch._check_caregiver_alerts(None)
        orch._check_caregiver_alerts(None)

        assert len(smtp.sent) == 1
        assert orch.notifier._already_notified == before

    def test_fresh_memory_not_alerted(self, emailing_orch, smtp):
        orch = emailing_orch
        orch.route(CRITICAL, agent_id="rajan")

        orch._check_caregiver_alerts(None)

        assert smtp.sent == []
        assert orch.notifier._already_notified == set()

    def test_non_critical_content_not_alerted(self, emailing_orch, smtp):
        orch = emailing_orch
        orch.route(HARMLESS, agent_id="rajan")
        age_to(orch, HARMLESS, 0.2)

        orch._check_caregiver_alerts(None)

        assert smtp.sent == []

    def test_escalation_warning_then_critical_notifies_twice(
        self, emailing_orch, smtp
    ):
        """Dedup key includes alert_level, so escalation re-notifies once."""
        orch = emailing_orch
        orch.route(CRITICAL, agent_id="rajan")

        ep = age_to(orch, CRITICAL, 0.45)
        assert 0.3 <= ep.retention() < 0.45
        orch._check_caregiver_alerts(None)
        age_to(orch, CRITICAL, 0.2)
        assert 0.1 < ep.retention() < 0.2
        orch._check_caregiver_alerts(None)
        orch._check_caregiver_alerts(None)

        assert len(smtp.sent) == 2
        assert "WARNING" in smtp.sent[0]
        assert "CRITICAL" in smtp.sent[1]
        assert {key(ep, "warning"), key(ep, "critical")} <= \
            orch.notifier._already_notified

    def test_new_critical_memory_later_is_notified(self, emailing_orch, smtp):
        orch = emailing_orch
        orch.route(CRITICAL, agent_id="rajan")
        age_to(orch, CRITICAL, 0.45)
        orch._check_caregiver_alerts(None)

        second = "my doctor appointment at the hospital is on monday"
        orch.route(second, agent_id="rajan")
        age_to(orch, second, 0.45)
        orch._check_caregiver_alerts(None)

        assert len(smtp.sent) == 2
        assert "hospital" in smtp.sent[1]

    @pytest.mark.parametrize(
        "follow_up, via",
        [
            # shares words with the memory → comes back as an Episode
            ("who brings my medicine every morning?", "episodic"),
        ],
    )
    def test_recalling_at_risk_memory_clears_alert(
        self, emailing_orch, smtp, follow_up, via
    ):
        """
        A fading memory that gets pulled into a later conversation is
        reinforced by the planner (last_reviewed_at reset, stability up),
        so it stops alerting and the caregiver isn't paged for it.
        """
        from memory.episodic_memory import Episode
        from memory.semantic_memory import SemanticMemory

        orch = emailing_orch
        orch.route(CRITICAL, agent_id="rajan")
        ep = age_to(orch, CRITICAL, 0.45)
        episodic = orch.memory_store.orchestrator.episodic_memory
        assert detect_alerts(episodic.get_all(), "rajan")

        result = orch.route(follow_up, agent_id="rajan")

        hit = next(m for m in result["planner_memories"]["fused_memory"]
                   if m.content == CRITICAL)
        assert isinstance(hit, Episode if via == "episodic"
                          else SemanticMemory)
        assert ep.retention() > 0.99
        assert detect_alerts(episodic.get_all(), "rajan") == []
        orch._check_caregiver_alerts(None)
        assert smtp.sent == []

    def test_unrelated_conversation_does_not_rescue_fading_memory(
        self, emailing_orch, smtp
    ):
        """
        Phase 9: a follow-up with nothing to do with the fading memory
        no longer drags it into the prompt (relevance cut-off), so it
        is not reinforced and the caregiver is still alerted — the
        forgetting curve isn't reset by unrelated chat.
        """
        orch = emailing_orch
        orch.route(CRITICAL, agent_id="rajan")
        ep = age_to(orch, CRITICAL, 0.45)

        result = orch.route("what a sunny afternoon outside", agent_id="rajan")

        assert CRITICAL not in [
            m.content for m in result["planner_memories"]["fused_memory"]
        ]
        assert ep.retention() < 0.5
        orch._check_caregiver_alerts(None)
        assert len(smtp.sent) == 1

    def test_only_monitored_agents_are_alerted(self, emailing_orch, smtp):
        orch = emailing_orch  # NOTIFY_AGENT_IDS=rajan
        orch.route(CRITICAL, agent_id="alice")
        age_to(orch, CRITICAL, 0.2, agent_id="alice")

        orch._check_caregiver_alerts(None)

        assert smtp.sent == []
        assert orch.notifier._already_notified == set()

    def test_alerts_do_not_cross_agents(self, real_orchestrator, smtp):
        orch = real_orchestrator(
            notify_agent_ids="rajan,alice", env=EMAIL_ENV
        )
        orch.route(CRITICAL, agent_id="alice")
        age_to(orch, CRITICAL, 0.45, agent_id="alice")

        orch._check_caregiver_alerts(None)

        assert len(smtp.sent) == 1
        assert "'alice'" in smtp.sent[0]
        assert all(k[0] == "alice" for k in orch.notifier._already_notified)

    def test_failed_send_is_retried_next_check(self, emailing_orch, smtp):
        orch = emailing_orch
        orch.route(CRITICAL, agent_id="rajan")
        ep = age_to(orch, CRITICAL, 0.45)

        smtp.fail = True
        orch._check_caregiver_alerts(None)
        assert orch.notifier._already_notified == set()

        smtp.fail = False
        orch._check_caregiver_alerts(None)
        assert len(smtp.sent) == 1
        assert key(ep, "warning") in orch.notifier._already_notified

    def test_log_only_mode_still_dedups(self, real_orchestrator, smtp):
        orch = real_orchestrator()  # no channel configured
        orch.route(CRITICAL, agent_id="rajan")
        ep = age_to(orch, CRITICAL, 0.45)

        orch._check_caregiver_alerts(None)
        orch._check_caregiver_alerts(None)

        assert orch.notifier._already_notified == {key(ep, "warning")}
        assert smtp.sent == []


# ─── Via the real scheduler cycle ──────────────────────────

class TestSchedulerPath:

    def test_scheduler_cycle_sends_warning(self, emailing_orch, smtp):
        orch = emailing_orch
        orch.route(CRITICAL, agent_id="rajan")
        ep = age_to(orch, CRITICAL, 0.45)

        orch.scheduler.run_once()
        orch.scheduler.run_once()

        assert len(smtp.sent) == 1
        assert key(ep, "warning") in orch.notifier._already_notified

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Real gap, needs a decision: _maintenance_cycle runs "
            "apply_decay()/cleanup() BEFORE the on_cycle_complete alert "
            "hook, and both delete memories with retention < 0.30 — the "
            "same line detect_alerts uses for 'critical'. So on the "
            "scheduled path a critical memory (e.g. medicine) is deleted "
            "and the caregiver is never told. Options: check alerts "
            "before decay, or exempt critical-keyword memories from "
            "pruning until alerted."
        ),
    )
    def test_scheduler_cycle_sends_critical_before_forgetting(
        self, emailing_orch, smtp
    ):
        orch = emailing_orch
        orch.route(CRITICAL, agent_id="rajan")
        ep = age_to(orch, CRITICAL, 0.2)

        orch.scheduler.run_once()

        assert key(ep, "critical") in orch.notifier._already_notified


# ─── Via POST /simulate_time ───────────────────────────────

class TestSimulateTimeEndpoint:

    @pytest.fixture
    def client(self, emailing_orch, monkeypatch):
        import dotenv
        import orchestrator.orchestrator as orch_module

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
        monkeypatch.setenv("MAEM_API_KEY", TEST_API_KEY)
        monkeypatch.setattr(orch_module, "Orchestrator",
                            lambda: emailing_orch)
        sys.modules.pop("api.main", None)
        module = importlib.import_module("api.main")
        yield TestClient(module.app)
        sys.modules.pop("api.main", None)

    def test_simulate_time_notifies_once(self, client, emailing_orch, smtp):
        h = {"X-API-Key": TEST_API_KEY}
        client.post("/chat", json={"message": CRITICAL, "user_id": "rajan"},
                    headers=h)
        ep = episode_with(emailing_orch, CRITICAL)
        hours = hours_to_retention(ep, 0.45)

        resp = client.post("/simulate_time",
                           json={"agent_id": "rajan", "hours": hours},
                           headers=h)
        assert resp.status_code == 200
        assert len(smtp.sent) == 1
        assert key(ep, "warning") in emailing_orch.notifier._already_notified

        client.post("/simulate_time",
                    json={"agent_id": "rajan", "hours": 0}, headers=h)
        assert len(smtp.sent) == 1

        alerts = client.get("/alerts/rajan", headers=h).json()
        assert alerts["alert_count"] == 1


# ─── Only the patient's own words raise alerts ─────────────

class TestAssistantRepliesDoNotAlert:
    """
    Found in the Phase 8 walkthrough: with an assistant that echoes or
    paraphrases the patient ("Noted: I take my insulin..."), the reply
    episode carries the same critical keywords and fades on its own
    schedule — so caregivers were paged about the AI's own words, for
    memories the dashboard panel doesn't even list.
    """

    def test_assistant_echo_never_alerts(self, emailing_orch, smtp):
        orch = emailing_orch
        orch.client.reply = lambda req: (
            "Noted: " + req["messages"][-1]["content"])
        orch.route(CRITICAL, agent_id="rajan")
        age_to(orch, CRITICAL, 0.45)

        alerts = detect_alerts(
            orch.memory_store.orchestrator.episodic_memory.get_all(),
            "rajan")
        orch._check_caregiver_alerts(None)

        assert [a["memory"] for a in alerts] == [CRITICAL]
        assert len(smtp.sent) == 1
        assert "Noted:" not in smtp.sent[0]

    def test_alerts_consistent_with_memory_panel(self, emailing_orch):
        orch = emailing_orch
        orch.client.reply = lambda req: (
            "Noted: " + req["messages"][-1]["content"])
        orch.route(CRITICAL, agent_id="rajan")
        age_to(orch, CRITICAL, 0.45)
        mo = orch.memory_store.orchestrator

        panel = {m["content"] for m in mo.metrics("rajan")["memories"]}
        alerted = {a["memory"] for a in
                   detect_alerts(mo.episodic_memory.get_all(), "rajan")}

        assert alerted <= panel


class TestNotificationDedupAcrossRestart:

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Needs a decision (found in the Phase 8 walkthrough): "
            "CaregiverNotifier._already_notified is in-memory only, so "
            "every process restart — including Hugging Face Spaces "
            "sleep/wake — re-sends every still-open alert at the startup "
            "maintenance cycle. Persisting notified keys (e.g. a table in "
            "the same SQLite DB) would fix it."
        ),
    )
    def test_restart_does_not_resend(self, real_orchestrator, smtp):
        first = real_orchestrator(env=EMAIL_ENV)
        first.route(CRITICAL, agent_id="rajan")
        age_to(first, CRITICAL, 0.45)
        first._check_caregiver_alerts(None)
        assert len(smtp.sent) == 1
        first.stop()

        second = real_orchestrator(env=EMAIL_ENV)  # same DB on disk
        second._check_caregiver_alerts(None)

        assert len(smtp.sent) == 1
