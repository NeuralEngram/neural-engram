# memory/notifier.py

"""
Caregiver Notification Module

Wraps the critical-memory detection that already backed the
GET /alerts/{agent_id} endpoint, and adds actual notification
channels (email and/or SMS) on top of it — no changes to the
memory tiers, agents, or orchestrator routing.

Usage
─────
detect_alerts(episodes, agent_id)  → same rule set /alerts used inline
CaregiverNotifier().notify(...)    → sends email/SMS for NEW alerts only,
                                      logs instead if nothing is configured
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Alert detection (moved here verbatim from api/main.py so both
# the /alerts endpoint and the scheduler hook share one definition)
# ─────────────────────────────────────────────────────────────

CRITICAL_KEYWORDS = [
    "medicine", "medication", "doctor",
    "appointment", "emergency", "hospital",
    "blood pressure", "insulin", "tablet",
    "daughter", "son", "family", "home"
]


def detect_alerts(
    episodes: List[Any],
    agent_id: str
) -> List[Dict[str, Any]]:
    """
    Scan episodes for critical-keyword memories whose retention
    has dropped below threshold. Same rule set previously inlined
    in api/main.py's /alerts/{agent_id} route.
    """

    alerts = []

    for ep in episodes:

        if ep.agent_id != agent_id:
            continue

        # Only the patient's own memories can trigger a caregiver alert.
        # The assistant's replies are stored as episodes too (role
        # "assistant"); without this, a reply that merely mentions
        # "medicine" or "doctor" paged the caregiver about the AI's own
        # words — while the memory panel (metrics(), which already
        # excludes assistant episodes) didn't even show that memory.
        context = ep.context if isinstance(ep.context, dict) else {}
        if context.get("role") == "assistant":
            continue

        retention = ep.retention()
        content_lower = str(ep.content).lower()

        is_critical = any(
            kw in content_lower
            for kw in CRITICAL_KEYWORDS
        )

        if is_critical and retention < 0.5:
            alerts.append({
                "memory_id": ep.episode_id,
                "memory": str(ep.content)[:80],
                "retention": round(retention, 3),
                "stability_hours": round(ep.stability_hours, 1),
                "alert_level": "critical" if retention < 0.3 else "warning"
            })

    return alerts


def _format_alert_body(agent_id: str, alerts: List[Dict[str, Any]]) -> str:
    lines = [f"Caregiver alert for patient '{agent_id}':", ""]
    for a in alerts:
        lines.append(
            f"- [{a['alert_level'].upper()}] \"{a['memory']}\" "
            f"(retention={a['retention']}, "
            f"stability={a['stability_hours']}h)"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Notifier
# ─────────────────────────────────────────────────────────────

class CaregiverNotifier:
    """
    Sends email and/or SMS notifications for caregiver alerts.

    Both channels are independently toggled and both are disabled
    by default. Enable email with ENABLE_EMAIL_ALERTS=true and the
    SMTP_* / ALERT_EMAIL_* env vars. Enable SMS with
    ENABLE_SMS_ALERTS=true and the TWILIO_* / ALERT_SMS_TO env vars
    (see .env.example). If a channel's config is incomplete, or the
    twilio package isn't installed, that channel falls back to
    log-only — this never breaks local dev, tests, or an
    unconfigured deploy.

    Keeps an in-memory record of which (agent_id, memory, level)
    combinations were already notified, so the same unresolved
    memory doesn't re-fire a notification every scheduler cycle.
    """

    def __init__(self):

        # ── Email config ──────────────────────────
        self.email_enabled = (
            os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
        )
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_addr = os.getenv("ALERT_EMAIL_FROM", self.smtp_user or "")
        self.to_addr = os.getenv("ALERT_EMAIL_TO")

        # SMTP itself is usable (host + login) — enough to email a
        # per-patient caregiver address even when ALERT_EMAIL_TO is unset.
        self.smtp_ready = self.email_enabled and all([
            self.smtp_host, self.smtp_user, self.smtp_password
        ])

        if self.email_enabled and not all([
            self.smtp_host, self.smtp_user,
            self.smtp_password, self.to_addr
        ]):
            logger.warning(
                "ENABLE_EMAIL_ALERTS=true but SMTP_HOST/SMTP_USER/"
                "SMTP_PASSWORD/ALERT_EMAIL_TO are not fully set — "
                "email alerts disabled, falling back to log-only."
            )
            self.email_enabled = False

        # ── SMS config (Twilio) ───────────────────
        self.sms_enabled = (
            os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true"
        )
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from = os.getenv("TWILIO_FROM_NUMBER")
        self.sms_to = os.getenv("ALERT_SMS_TO")

        if self.sms_enabled and not all([
            self.twilio_sid, self.twilio_token,
            self.twilio_from, self.sms_to
        ]):
            logger.warning(
                "ENABLE_SMS_ALERTS=true but TWILIO_ACCOUNT_SID/"
                "TWILIO_AUTH_TOKEN/TWILIO_FROM_NUMBER/ALERT_SMS_TO "
                "are not fully set — SMS alerts disabled, "
                "falling back to log-only."
            )
            self.sms_enabled = False

        if self.sms_enabled:
            try:
                from twilio.rest import Client  # noqa: F401
            except ImportError:
                logger.warning(
                    "ENABLE_SMS_ALERTS=true but the 'twilio' package "
                    "isn't installed (pip install twilio) — SMS "
                    "alerts disabled, falling back to log-only."
                )
                self.sms_enabled = False

        self._already_notified = set()

    # ─────────────────────────────────────────────────────
    # Dedup bookkeeping
    # ─────────────────────────────────────────────────────

    def _new_alerts(
        self,
        agent_id: str,
        alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        return [
            a for a in alerts
            if (agent_id, a["memory"], a["alert_level"])
            not in self._already_notified
        ]

    def _mark_notified(
        self,
        agent_id: str,
        alerts: List[Dict[str, Any]]
    ):

        for a in alerts:
            self._already_notified.add(
                (agent_id, a["memory"], a["alert_level"])
            )

    # ─────────────────────────────────────────────────────
    # Channel senders
    # ─────────────────────────────────────────────────────

    def _send_email(
        self,
        agent_id: str,
        alerts: List[Dict[str, Any]],
        to_addrs: Optional[List[str]] = None
    ) -> bool:

        recipients = list(to_addrs) if to_addrs else [self.to_addr]
        body = _format_alert_body(agent_id, alerts)

        message = MIMEText(body)
        message["Subject"] = (
            f"MAEM Caregiver Alert — {agent_id} "
            f"({len(alerts)} item(s))"
        )
        message["From"] = self.from_addr or self.smtp_user
        message["To"] = ", ".join(recipients)

        try:
            with smtplib.SMTP(
                self.smtp_host, self.smtp_port, timeout=10
            ) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(
                    self.from_addr or self.smtp_user, recipients,
                    message.as_string()
                )

            logger.info(
                "Sent caregiver alert email for %s (%d item(s))",
                agent_id, len(alerts)
            )
            return True

        except Exception:
            logger.exception(
                "Failed to send caregiver alert email for %s", agent_id
            )
            return False

    def _send_sms(self, agent_id: str, alerts: List[Dict[str, Any]]) -> bool:

        from twilio.rest import Client

        body = _format_alert_body(agent_id, alerts)

        # SMS has a practical length limit — keep it short and
        # point to the dashboard for full detail.
        if len(body) > 300:
            body = body[:297] + "..."

        try:
            client = Client(self.twilio_sid, self.twilio_token)
            client.messages.create(
                body=body,
                from_=self.twilio_from,
                to=self.sms_to
            )

            logger.info(
                "Sent caregiver alert SMS for %s (%d item(s))",
                agent_id, len(alerts)
            )
            return True

        except Exception:
            logger.exception(
                "Failed to send caregiver alert SMS for %s", agent_id
            )
            return False

    # ─────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────

    def notify(
        self,
        agent_id: str,
        alerts: List[Dict[str, Any]],
        to_addrs: Optional[List[str]] = None
    ) -> int:
        """
        Notify caregivers of any alert not already sent, via every
        enabled channel. Returns the number of alerts that were
        newly notified (through at least one channel, or logged if
        no channel is configured).
        """

        new_alerts = self._new_alerts(agent_id, alerts)

        if not new_alerts:
            return 0

        # Per-patient caregiver address (application accounts): send
        # there whenever SMTP is configured, even without ALERT_EMAIL_TO.
        if to_addrs and self.smtp_ready:
            if self._send_email(agent_id, new_alerts, to_addrs):
                self._mark_notified(agent_id, new_alerts)
                return len(new_alerts)
            return 0

        if not self.email_enabled and not self.sms_enabled:
            logger.info(
                "Caregiver alert(s) for %s (no channel enabled — "
                "log only): %s",
                agent_id, new_alerts
            )
            self._mark_notified(agent_id, new_alerts)
            return len(new_alerts)

        sent_any = False

        if self.email_enabled:
            if self._send_email(agent_id, new_alerts):
                sent_any = True

        if self.sms_enabled:
            if self._send_sms(agent_id, new_alerts):
                sent_any = True

        if not sent_any:
            # Both enabled channels failed at send time — don't
            # mark as notified so the next cycle retries.
            return 0

        self._mark_notified(agent_id, new_alerts)
        return len(new_alerts)