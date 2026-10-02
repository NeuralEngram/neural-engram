# accounts/store.py

"""
SQLite-backed account store.

Tables
──────
users                  username, display name, password hash, role
                       ("user" | "caregiver"), account type ("standard" |
                       "alzheimers"), severity, decay multiplier
caregiver_links        caregiver → Alzheimer's patient (one caregiver per
                       patient). Standard users can never be linked.
reminders              tasks / reminders for a user, created by the user
                       or by their caregiver
alerts                 caregiver alerts (open / handled / resolved) with
                       history
retention_snapshots    one row per user per day: average retention etc.
notification_settings  per caregiver: email alerts on/off + address

The database lives next to the episodic memory database (same folder)
unless MAEM_APP_DB_PATH says otherwise, so test runs and evaluation runs
that redirect the memory DB to a temporary folder get a temporary
accounts DB too.
"""

import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from accounts.security import hash_password, verify_password

ROLE_USER = "user"
ROLE_CAREGIVER = "caregiver"
TYPE_STANDARD = "standard"
TYPE_ALZHEIMERS = "alzheimers"

# Decay speed by severity for newly registered Alzheimer's users.
SEVERITY_DECAY = {"mild": 1.5, "moderate": 3.0, "severe": 5.0}

USERNAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,19}$")

SEED_STANDARD = ["Arjun", "Sid", "Grace", "Ivy", "Chris"]
SEED_ALZHEIMERS = [
    # name, severity, decay multiplier
    ("Will", "mild", 1.5),
    ("Henry", "mild", 2.0),
    ("Siri", "moderate", 3.0),
    ("Zoya", "moderate", 4.0),
    ("Riko", "severe", 5.0),
]
SEED_CAREGIVERS = {
    "Siya": ["Will", "Henry", "Siri"],
    "Ansh": ["Zoya", "Riko"],
}


def seed_password(name: str) -> str:
    return f"{name.lower()}@123"


class AccountError(ValueError):
    """Invalid account operation (shown to the user as a 400)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_db_path() -> str:
    raw = os.getenv("MAEM_APP_DB_PATH", "").strip()
    if raw:
        return raw if os.path.isabs(raw) else os.path.join(_project_root(), raw)
    try:
        import memory.episodic_memory as episodic_module
        folder = os.path.dirname(os.path.abspath(episodic_module.DB_PATH))
    except Exception:
        folder = _project_root()
    return os.path.join(folder, "maem_app.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL,
    account_type    TEXT,
    severity        TEXT,
    decay_multiplier REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS caregiver_links (
    caregiver   TEXT NOT NULL,
    patient     TEXT NOT NULL UNIQUE,
    PRIMARY KEY (caregiver, patient)
);
CREATE TABLE IF NOT EXISTS reminders (
    id          TEXT PRIMARY KEY,
    username    TEXT NOT NULL,
    title       TEXT NOT NULL,
    due_at      TEXT,
    done        INTEGER NOT NULL DEFAULT 0,
    created_by  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id          TEXT PRIMARY KEY,
    username    TEXT NOT NULL,
    memory_id   TEXT NOT NULL,
    memory      TEXT NOT NULL,
    level       TEXT NOT NULL,
    retention   REAL NOT NULL,
    status      TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    handled_at  TEXT,
    handled_by  TEXT,
    UNIQUE (username, memory_id, level)
);
CREATE TABLE IF NOT EXISTS retention_snapshots (
    username      TEXT NOT NULL,
    day           TEXT NOT NULL,
    avg_retention REAL NOT NULL,
    memory_count  INTEGER NOT NULL,
    at_risk       INTEGER NOT NULL,
    PRIMARY KEY (username, day)
);
CREATE TABLE IF NOT EXISTS notification_settings (
    caregiver     TEXT PRIMARY KEY,
    email_enabled INTEGER NOT NULL DEFAULT 0,
    email         TEXT
);
"""


class AccountStore:

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or default_db_path()
        folder = os.path.dirname(self.db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ─────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────

    def _q(self, sql: str, params: Iterable = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def _x(self, sql: str, params: Iterable = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur.rowcount

    @staticmethod
    def _user_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "username": row["username"],
            "name": row["name"],
            "role": row["role"],
            "account_type": row["account_type"],
            "severity": row["severity"],
            "decay_multiplier": row["decay_multiplier"],
            "created_at": row["created_at"],
        }

    def close(self):
        with self._lock:
            self._conn.close()

    # ─────────────────────────────────────────────────────
    # users
    # ─────────────────────────────────────────────────────

    def user_count(self) -> int:
        return self._q("SELECT COUNT(*) AS n FROM users")[0]["n"]

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        rows = self._q("SELECT * FROM users WHERE username = ?",
                       (username.lower(),))
        return self._user_dict(rows[0]) if rows else None

    def list_users(self) -> List[Dict[str, Any]]:
        return [self._user_dict(r) for r in
                self._q("SELECT * FROM users ORDER BY role, name")]

    def create_user(
        self,
        name: str,
        username: str,
        password: str,
        account_type: str,
        severity: Optional[str] = None,
        caregiver: Optional[str] = None,
        decay_multiplier: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        account_type: "standard" | "alzheimers" | "caregiver".
        Alzheimer's users need a severity and an existing caregiver;
        standard users and caregivers must not name a caregiver.
        """
        name = (name or "").strip()
        username = (username or "").strip().lower()
        if not name or len(name) > 40:
            raise AccountError("Name is required (max 40 characters).")
        if not USERNAME_RE.match(username):
            raise AccountError(
                "Username must be 3–20 characters: letters, numbers or _, "
                "starting with a letter.")
        if len(password or "") < 6:
            raise AccountError("Password must be at least 6 characters.")
        if self.get_user(username):
            raise AccountError("That username is already taken.")

        if account_type == ROLE_CAREGIVER:
            if caregiver:
                raise AccountError("A caregiver account can't have a caregiver.")
            role, acct, sev, decay = ROLE_CAREGIVER, None, None, 1.0
        elif account_type == TYPE_STANDARD:
            if caregiver:
                raise AccountError(
                    "Standard accounts can't be linked to a caregiver.")
            role, acct, sev, decay = ROLE_USER, TYPE_STANDARD, None, 1.0
        elif account_type == TYPE_ALZHEIMERS:
            if severity not in SEVERITY_DECAY:
                raise AccountError(
                    "Choose a severity: mild, moderate or severe.")
            cg = self.get_user(caregiver or "")
            if not cg or cg["role"] != ROLE_CAREGIVER:
                raise AccountError(
                    "An Alzheimer's account needs an existing caregiver.")
            role, acct, sev = ROLE_USER, TYPE_ALZHEIMERS, severity
            decay = decay_multiplier or SEVERITY_DECAY[severity]
        else:
            raise AccountError(
                "Account type must be standard, alzheimers or caregiver.")

        with self._lock:
            self._conn.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?,?,?)",
                (username, name, hash_password(password), role, acct, sev,
                 float(decay), _now()))
            if acct == TYPE_ALZHEIMERS:
                self._conn.execute(
                    "INSERT INTO caregiver_links VALUES (?,?)",
                    (caregiver.lower(), username))
            if role == ROLE_CAREGIVER:
                self._conn.execute(
                    "INSERT OR IGNORE INTO notification_settings "
                    "VALUES (?,0,NULL)", (username,))
            self._conn.commit()
        return self.get_user(username)

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        rows = self._q("SELECT * FROM users WHERE username = ?",
                       ((username or "").strip().lower(),))
        if not rows or not verify_password(password or "", rows[0]["password_hash"]):
            return None
        return self._user_dict(rows[0])

    def change_password(self, username: str, old: str, new: str) -> None:
        if not self.authenticate(username, old):
            raise AccountError("Current password is incorrect.")
        if len(new or "") < 6:
            raise AccountError("New password must be at least 6 characters.")
        self._x("UPDATE users SET password_hash = ? WHERE username = ?",
                (hash_password(new), username))

    def ensure_seeded(self) -> bool:
        """Create the default accounts on first run. Returns True if seeded."""
        if self.user_count() > 0:
            return False
        for cg in SEED_CAREGIVERS:
            self.create_user(cg, cg.lower(), seed_password(cg), ROLE_CAREGIVER)
        caregiver_of = {p: cg for cg, ps in SEED_CAREGIVERS.items() for p in ps}
        for name in SEED_STANDARD:
            self.create_user(name, name.lower(), seed_password(name),
                             TYPE_STANDARD)
        for name, severity, decay in SEED_ALZHEIMERS:
            self.create_user(name, name.lower(), seed_password(name),
                             TYPE_ALZHEIMERS, severity=severity,
                             caregiver=caregiver_of[name].lower(),
                             decay_multiplier=decay)
        return True

    def decay_multipliers(self) -> Dict[str, float]:
        return {r["username"]: r["decay_multiplier"] for r in self._q(
            "SELECT username, decay_multiplier FROM users WHERE role = ?",
            (ROLE_USER,))}

    def alzheimers_usernames(self) -> List[str]:
        return [r["username"] for r in self._q(
            "SELECT username FROM users WHERE account_type = ?",
            (TYPE_ALZHEIMERS,))]

    def member_usernames(self) -> List[str]:
        return [r["username"] for r in self._q(
            "SELECT username FROM users WHERE role = ?", (ROLE_USER,))]

    # ─────────────────────────────────────────────────────
    # caregiver links
    # ─────────────────────────────────────────────────────

    def patients_of(self, caregiver: str) -> List[Dict[str, Any]]:
        rows = self._q(
            "SELECT u.* FROM users u JOIN caregiver_links l "
            "ON u.username = l.patient WHERE l.caregiver = ? ORDER BY u.name",
            (caregiver,))
        return [self._user_dict(r) for r in rows]

    def caregiver_of(self, patient: str) -> Optional[str]:
        rows = self._q("SELECT caregiver FROM caregiver_links WHERE patient = ?",
                       (patient,))
        return rows[0]["caregiver"] if rows else None

    def is_caregiver_of(self, caregiver: str, patient: str) -> bool:
        return self.caregiver_of(patient) == caregiver

    def can_access(self, viewer: Dict[str, Any], target: str) -> bool:
        """A user sees only their own data; a caregiver only their patients."""
        if viewer["role"] == ROLE_CAREGIVER:
            return self.is_caregiver_of(viewer["username"], target)
        return viewer["username"] == target

    # ─────────────────────────────────────────────────────
    # reminders
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _reminder_dict(r: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": r["id"], "username": r["username"], "title": r["title"],
            "due_at": r["due_at"], "done": bool(r["done"]),
            "created_by": r["created_by"], "created_at": r["created_at"],
        }

    def list_reminders(self, username: str) -> List[Dict[str, Any]]:
        rows = self._q(
            "SELECT * FROM reminders WHERE username = ? "
            "ORDER BY done, CASE WHEN due_at IS NULL THEN 1 ELSE 0 END, "
            "due_at, created_at", (username,))
        return [self._reminder_dict(r) for r in rows]

    def add_reminder(self, username: str, title: str,
                     due_at: Optional[str], created_by: str) -> Dict[str, Any]:
        title = (title or "").strip()
        if not title or len(title) > 200:
            raise AccountError("Reminder text is required (max 200 characters).")
        if due_at:
            try:
                datetime.fromisoformat(due_at)
            except ValueError as exc:
                raise AccountError("Invalid reminder date/time.") from exc
        rid = str(uuid.uuid4())
        self._x("INSERT INTO reminders VALUES (?,?,?,?,0,?,?)",
                (rid, username, title, due_at or None, created_by, _now()))
        return self.get_reminder(rid)

    def get_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        rows = self._q("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        return self._reminder_dict(rows[0]) if rows else None

    def set_reminder_done(self, reminder_id: str, done: bool) -> None:
        self._x("UPDATE reminders SET done = ? WHERE id = ?",
                (1 if done else 0, reminder_id))

    def delete_reminder(self, reminder_id: str) -> None:
        self._x("DELETE FROM reminders WHERE id = ?", (reminder_id,))

    # ─────────────────────────────────────────────────────
    # alerts
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _alert_dict(r: sqlite3.Row) -> Dict[str, Any]:
        return {k: r[k] for k in r.keys()}

    def sync_alerts(self, username: str, current: List[Dict[str, Any]]) -> None:
        """
        Record the alerts that are active right now for `username`.
        New ones are opened; ones no longer active (memory reinforced or
        forgotten and pruned) are marked resolved.
        """
        now = _now()
        active = set()
        with self._lock:
            for a in current:
                mid = a.get("memory_id") or a["memory"]
                active.add((mid, a["alert_level"]))
                row = self._conn.execute(
                    "SELECT id, status FROM alerts WHERE username=? AND "
                    "memory_id=? AND level=?",
                    (username, mid, a["alert_level"])).fetchone()
                if row is None:
                    self._conn.execute(
                        "INSERT INTO alerts VALUES (?,?,?,?,?,?,'open',?,?,NULL,NULL)",
                        (str(uuid.uuid4()), username, mid, a["memory"],
                         a["alert_level"], float(a["retention"]), now, now))
                else:
                    status = "open" if row["status"] == "resolved" else row["status"]
                    self._conn.execute(
                        "UPDATE alerts SET retention=?, last_seen=?, status=? "
                        "WHERE id=?",
                        (float(a["retention"]), now, status, row["id"]))
            for row in self._conn.execute(
                    "SELECT id, memory_id, level FROM alerts WHERE username=? "
                    "AND status='open'", (username,)).fetchall():
                if (row["memory_id"], row["level"]) not in active:
                    self._conn.execute(
                        "UPDATE alerts SET status='resolved' WHERE id=?",
                        (row["id"],))
            self._conn.commit()

    def list_alerts(self, username: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            rows = self._q("SELECT * FROM alerts WHERE username=? AND status=? "
                           "ORDER BY last_seen DESC", (username, status))
        else:
            rows = self._q("SELECT * FROM alerts WHERE username=? "
                           "ORDER BY last_seen DESC", (username,))
        return [self._alert_dict(r) for r in rows]

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        rows = self._q("SELECT * FROM alerts WHERE id=?", (alert_id,))
        return self._alert_dict(rows[0]) if rows else None

    def handle_alert(self, alert_id: str, by: str) -> None:
        self._x("UPDATE alerts SET status='handled', handled_at=?, handled_by=? "
                "WHERE id=?", (_now(), by, alert_id))

    def open_alert_count(self, username: str) -> int:
        return self._q("SELECT COUNT(*) AS n FROM alerts WHERE username=? "
                       "AND status='open'", (username,))[0]["n"]

    # ─────────────────────────────────────────────────────
    # retention snapshots
    # ─────────────────────────────────────────────────────

    def record_snapshot(self, username: str, avg_retention: float,
                        memory_count: int, at_risk: int,
                        day: Optional[str] = None) -> None:
        day = day or datetime.now().date().isoformat()
        self._x("INSERT OR REPLACE INTO retention_snapshots VALUES (?,?,?,?,?)",
                (username, day, float(avg_retention), int(memory_count),
                 int(at_risk)))

    def snapshots(self, username: str, days: int = 14) -> List[Dict[str, Any]]:
        rows = self._q("SELECT * FROM retention_snapshots WHERE username=? "
                       "ORDER BY day DESC LIMIT ?", (username, int(days)))
        return [dict(r) for r in reversed(rows)]

    # ─────────────────────────────────────────────────────
    # notification settings
    # ─────────────────────────────────────────────────────

    def get_notifications(self, caregiver: str) -> Dict[str, Any]:
        rows = self._q("SELECT * FROM notification_settings WHERE caregiver=?",
                       (caregiver,))
        if not rows:
            return {"email_enabled": False, "email": None}
        return {"email_enabled": bool(rows[0]["email_enabled"]),
                "email": rows[0]["email"]}

    def set_notifications(self, caregiver: str, email_enabled: bool,
                          email: Optional[str]) -> Dict[str, Any]:
        email = (email or "").strip() or None
        if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            raise AccountError("Enter a valid email address.")
        if email_enabled and not email:
            raise AccountError("Add an email address to turn on email alerts.")
        self._x("INSERT OR REPLACE INTO notification_settings VALUES (?,?,?)",
                (caregiver, 1 if email_enabled else 0, email))
        return self.get_notifications(caregiver)

    def alert_recipients(self, patient: str) -> List[str]:
        """Email addresses of the patient's caregiver, if they opted in."""
        cg = self.caregiver_of(patient)
        if not cg:
            return []
        s = self.get_notifications(cg)
        return [s["email"]] if s["email_enabled"] and s["email"] else []

    # ─────────────────────────────────────────────────────
    # reset
    # ─────────────────────────────────────────────────────

    def delete_user_data(self, username: str) -> None:
        with self._lock:
            for table in ("reminders", "alerts", "retention_snapshots"):
                self._conn.execute(f"DELETE FROM {table} WHERE username=?",
                                   (username,))
            self._conn.commit()

    def delete_all_data(self) -> None:
        with self._lock:
            for table in ("reminders", "alerts", "retention_snapshots"):
                self._conn.execute(f"DELETE FROM {table}")
            self._conn.commit()
