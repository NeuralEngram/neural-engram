"""
Phase 10 — application API (/app): accounts, roles and isolation.

Seeded accounts (created on first startup):
  standard users  arjun sid grace ivy chris
  Alzheimer's     will henry (mild) siri zoya (moderate) riko (severe)
  caregivers      siya → will, henry, siri    ansh → zoya, riko
  password        <username>@123
"""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TEST_API_KEY


@pytest.fixture
def app_client(api_app):
    with TestClient(api_app.app) as c:   # `with` runs startup → seeding
        yield c


def login(client, username, password=None):
    r = client.post("/app/auth/login", headers={"X-API-Key": TEST_API_KEY},
                    json={"username": username,
                          "password": password or f"{username}@123"})
    assert r.status_code == 200, r.text
    return {"X-API-Key": TEST_API_KEY,
            "Authorization": f"Bearer {r.json()['token']}"}


# ─── Access layers ─────────────────────────────────────────

class TestAccess:

    def test_api_key_required_even_for_login(self, app_client):
        r = app_client.post("/app/auth/login",
                            json={"username": "will", "password": "will@123"})
        assert r.status_code == 401

    def test_login_required_for_data(self, app_client):
        r = app_client.get("/app/memories",
                           headers={"X-API-Key": TEST_API_KEY})
        assert r.status_code == 401

    def test_wrong_password(self, app_client):
        r = app_client.post("/app/auth/login",
                            headers={"X-API-Key": TEST_API_KEY},
                            json={"username": "will", "password": "nope"})
        assert r.status_code == 401

    def test_garbage_token_rejected(self, app_client):
        r = app_client.get("/app/auth/me", headers={
            "X-API-Key": TEST_API_KEY, "Authorization": "Bearer abc.def.ghi"})
        assert r.status_code == 401

    def test_login_is_case_insensitive_on_username(self, app_client):
        login(app_client, "Will", "will@123")


# ─── Seeded accounts ───────────────────────────────────────

class TestSeed:

    def test_standard_user_has_no_caregiver(self, app_client):
        me = app_client.get("/app/auth/me", headers=login(app_client, "arjun")).json()
        assert me["role"] == "user"
        assert me["account_type"] == "standard"
        assert "caregiver" not in me

    def test_alzheimers_user_profile(self, app_client):
        me = app_client.get("/app/auth/me", headers=login(app_client, "riko")).json()
        assert me["account_type"] == "alzheimers"
        assert me["severity"] == "severe"
        assert me["decay_multiplier"] == 5.0
        assert me["caregiver"]["username"] == "ansh"

    @pytest.mark.parametrize("cg, patients", [
        ("siya", ["henry", "siri", "will"]),
        ("ansh", ["riko", "zoya"]),
    ])
    def test_caregiver_patients(self, app_client, cg, patients):
        rows = app_client.get("/app/patients", headers=login(app_client, cg)).json()
        assert sorted(p["username"] for p in rows) == patients


# ─── Isolation ─────────────────────────────────────────────

class TestIsolation:

    @pytest.mark.parametrize("viewer, target", [
        ("arjun", "will"),   # user → someone else
        ("will", "henry"),   # patient → another patient
        ("siya", "zoya"),    # caregiver → other caregiver's patient
        ("siya", "arjun"),   # caregiver → standard user (never allowed)
        ("ansh", "sid"),
    ])
    def test_cannot_read_other_accounts(self, app_client, viewer, target):
        h = login(app_client, viewer)
        for path in ("/app/memories", "/app/reminders", "/app/today",
                     "/app/trend"):
            r = app_client.get(path, params={"user": target}, headers=h)
            assert r.status_code == 403, (path, r.text)

    def test_caregiver_can_read_own_patient(self, app_client):
        r = app_client.get("/app/memories", params={"user": "will"},
                           headers=login(app_client, "siya"))
        assert r.status_code == 200

    def test_caregiver_must_pick_a_patient(self, app_client):
        r = app_client.get("/app/memories", headers=login(app_client, "siya"))
        assert r.status_code == 400

    def test_caregivers_cannot_chat(self, app_client):
        r = app_client.post("/app/chat", json={"message": "hi"},
                            headers=login(app_client, "siya"))
        assert r.status_code == 403

    def test_users_cannot_use_caregiver_pages(self, app_client):
        h = login(app_client, "will")
        assert app_client.get("/app/patients", headers=h).status_code == 403
        assert app_client.get("/app/alerts", params={"user": "will"},
                              headers=h).status_code == 403


# ─── Registration rules ────────────────────────────────────

class TestRegister:

    def reg(self, client, **body):
        return client.post("/app/auth/register",
                           headers={"X-API-Key": TEST_API_KEY}, json=body)

    def test_standard_cannot_have_caregiver(self, app_client):
        r = self.reg(app_client, name="Tom", username="tom", password="secret1",
                     account_type="standard", caregiver="siya")
        assert r.status_code == 400

    def test_alzheimers_needs_caregiver(self, app_client):
        r = self.reg(app_client, name="Tom", username="tom", password="secret1",
                     account_type="alzheimers", severity="mild")
        assert r.status_code == 400

    def test_alzheimers_caregiver_must_be_a_caregiver(self, app_client):
        r = self.reg(app_client, name="Tom", username="tom", password="secret1",
                     account_type="alzheimers", severity="mild",
                     caregiver="arjun")
        assert r.status_code == 400

    def test_duplicate_username(self, app_client):
        r = self.reg(app_client, name="Will", username="will",
                     password="secret1", account_type="standard")
        assert r.status_code == 400

    def test_new_patient_linked_to_caregiver(self, app_client):
        r = self.reg(app_client, name="Tom", username="tom", password="secret1",
                     account_type="alzheimers", severity="moderate",
                     caregiver="ansh")
        assert r.status_code == 200
        assert r.json()["user"]["decay_multiplier"] == 3.0
        rows = app_client.get("/app/patients", headers=login(app_client, "ansh")).json()
        assert "tom" in [p["username"] for p in rows]

    def test_caregiver_list_for_form(self, app_client):
        r = app_client.get("/app/auth/caregivers",
                           headers={"X-API-Key": TEST_API_KEY})
        assert sorted(c["username"] for c in r.json()) == ["ansh", "siya"]


# ─── Memories ──────────────────────────────────────────────

class TestMemories:

    def test_chat_creates_memory_with_full_fields(self, app_client):
        h = login(app_client, "arjun")
        r = app_client.post("/app/chat", json={"message": "I live in Jayanagar"},
                            headers=h)
        assert r.status_code == 200
        mem = app_client.get("/app/memories", headers=h).json()
        item = mem["memories"][0]
        assert item["content"] == "I live in Jayanagar"
        for key in ("retention", "importance", "priority", "times_retrieved",
                    "stability_hours", "health", "created_at", "last_recalled"):
            assert key in item
        assert mem["summary"]["memory_count"] == 1

    def test_bot_replies_not_listed(self, app_client):
        h = login(app_client, "arjun")
        app_client.post("/app/chat", json={"message": "I live in Jayanagar"},
                        headers=h)
        contents = [m["content"] for m in
                    app_client.get("/app/memories", headers=h).json()["memories"]]
        assert "Okay, noted." not in contents

    def test_alzheimers_memory_decays_faster(self, app_client):
        h = login(app_client, "riko")
        app_client.post("/app/memories", json={"text": "my keys are in the drawer"},
                        headers=h)
        h2 = login(app_client, "arjun")
        app_client.post("/app/memories", json={"text": "my keys are in the drawer"},
                        headers=h2)
        for user, hdr in (("riko", h), ("arjun", h2)):
            app_client.post("/app/tools/simulate", json={"hours": 24}, headers=hdr)
        riko = app_client.get("/app/memories", headers=h).json()["memories"][0]
        arjun = app_client.get("/app/memories", headers=h2).json()["memories"][0]
        assert riko["retention"] < arjun["retention"]

    def test_caregiver_adds_important_memory(self, app_client):
        h = login(app_client, "siya")
        r = app_client.post("/app/memories", headers=h, json={
            "text": "Doctor appointment on Friday at 10 am",
            "important": True, "user": "will"})
        assert r.status_code == 200
        assert r.json()["important"] is True
        assert r.json()["source"] == "caregiver"
        assert r.json()["stability_hours"] >= 67

    def test_reinforce(self, app_client):
        h = login(app_client, "will")
        mid = app_client.post("/app/memories", json={"text": "my son visits"},
                              headers=h).json()["id"]
        r = app_client.post(f"/app/memories/{mid}/reinforce", headers=h)
        assert r.json()["times_retrieved"] == 1
        other = login(app_client, "arjun")
        assert app_client.post(f"/app/memories/{mid}/reinforce",
                               headers=other).status_code == 403


# ─── Reminders & Today ─────────────────────────────────────

class TestReminders:

    def test_caregiver_sets_reminder_patient_sees_it(self, app_client):
        siya = login(app_client, "siya")
        due = datetime.now().replace(microsecond=0).isoformat()
        r = app_client.post("/app/reminders", headers=siya, json={
            "title": "Take evening tablet", "due_at": due, "user": "will"})
        assert r.status_code == 200
        rid = r.json()["id"]

        will = login(app_client, "will")
        rows = app_client.get("/app/reminders", headers=will).json()
        assert rows[0]["title"] == "Take evening tablet"
        assert rows[0]["created_by"] == "siya"

        today = app_client.get("/app/today", headers=will).json()
        assert [x["id"] for x in today["reminders"]] == [rid]

        assert app_client.patch(f"/app/reminders/{rid}", json={"done": True},
                                headers=will).json()["done"] is True
        arjun = login(app_client, "arjun")
        assert app_client.patch(f"/app/reminders/{rid}", json={"done": False},
                                headers=arjun).status_code == 403

    def test_future_reminder_not_in_today(self, app_client):
        h = login(app_client, "arjun")
        later = (datetime.now() + timedelta(days=3)).isoformat()
        app_client.post("/app/reminders", json={"title": "Dentist", "due_at": later},
                        headers=h)
        assert app_client.get("/app/today", headers=h).json()["reminders"] == []

    def test_bad_date_rejected(self, app_client):
        r = app_client.post("/app/reminders", json={"title": "x", "due_at": "soon"},
                            headers=login(app_client, "arjun"))
        assert r.status_code == 400

    def test_gentle_reminders_only_for_alzheimers(self, app_client):
        for user in ("riko", "arjun"):
            h = login(app_client, user)
            app_client.post("/app/memories", headers=h,
                            json={"text": "I take my medicine after breakfast"})
            # riko (×5 decay): retention ≈ 0.52 → fading, not yet forgotten
            app_client.post("/app/tools/simulate", json={"hours": 6}, headers=h)
        riko = app_client.get("/app/today", headers=login(app_client, "riko")).json()
        arjun = app_client.get("/app/today", headers=login(app_client, "arjun")).json()
        assert riko["gentle_reminders"]
        assert arjun["gentle_reminders"] == []


# ─── Alerts ────────────────────────────────────────────────

class TestAlerts:

    def test_open_handle_history(self, app_client):
        riko = login(app_client, "riko")
        app_client.post("/app/memories", headers=riko,
                        json={"text": "I take insulin before dinner"})
        app_client.post("/app/tools/simulate", json={"hours": 30}, headers=riko)

        ansh = login(app_client, "ansh")
        open_alerts = app_client.get("/app/alerts", params={"user": "riko",
                                     "status": "open"}, headers=ansh).json()
        assert open_alerts and open_alerts[0]["status"] == "open"

        aid = open_alerts[0]["id"]
        r = app_client.post(f"/app/alerts/{aid}/handle", headers=ansh)
        assert r.json()["status"] == "handled"
        assert r.json()["handled_by"] == "ansh"

        history = app_client.get("/app/alerts", params={"user": "riko"},
                                 headers=ansh).json()
        assert any(a["id"] == aid for a in history)

        siya = login(app_client, "siya")
        assert app_client.post(f"/app/alerts/{aid}/handle",
                               headers=siya).status_code == 403

    def test_alerts_resolve_when_memory_reinforced(self, app_client):
        riko = login(app_client, "riko")
        mid = app_client.post("/app/memories", headers=riko,
                              json={"text": "my doctor is Dr. Mehta"}).json()["id"]
        app_client.post("/app/tools/simulate", json={"hours": 30}, headers=riko)
        ansh = login(app_client, "ansh")
        assert app_client.get("/app/alerts", params={"user": "riko",
                              "status": "open"}, headers=ansh).json()
        app_client.post(f"/app/memories/{mid}/reinforce", headers=riko)
        assert app_client.get("/app/alerts", params={"user": "riko",
                              "status": "open"}, headers=ansh).json() == []


# ─── Settings, tools, reset ────────────────────────────────

class TestSettingsAndTools:

    def test_notification_settings(self, app_client):
        siya = login(app_client, "siya")
        bad = app_client.put("/app/settings/notifications", headers=siya,
                             json={"email_enabled": True, "email": ""})
        assert bad.status_code == 400
        ok = app_client.put("/app/settings/notifications", headers=siya,
                            json={"email_enabled": True, "email": "siya@example.com"})
        assert ok.json() == {"email_enabled": True, "email": "siya@example.com"}
        assert app_client.get("/app/settings/notifications",
                              headers=login(app_client, "arjun")).status_code == 403

    def test_compare_shows_only_visible_accounts(self, app_client):
        arjun = app_client.get("/app/tools/compare",
                               headers=login(app_client, "arjun")).json()
        assert [a["username"] for a in arjun["accounts"]] == ["arjun"]
        assert len(arjun["curves"]) == 4
        siya = app_client.get("/app/tools/compare",
                              headers=login(app_client, "siya")).json()
        assert sorted(a["username"] for a in siya["accounts"]) == [
            "henry", "siri", "will"]

    def test_trend_has_today(self, app_client):
        h = login(app_client, "arjun")
        app_client.post("/app/memories", json={"text": "I like cricket"}, headers=h)
        rows = app_client.get("/app/trend", headers=h).json()
        assert rows[-1]["day"] == datetime.now().date().isoformat()

    def test_change_password(self, app_client):
        h = login(app_client, "grace")
        assert app_client.post("/app/auth/password", headers=h, json={
            "current_password": "wrong", "new_password": "newpass1"}
        ).status_code == 400
        assert app_client.post("/app/auth/password", headers=h, json={
            "current_password": "grace@123", "new_password": "newpass1"}
        ).status_code == 200
        login(app_client, "grace", "newpass1")

    def test_reset_own_memories(self, app_client):
        h = login(app_client, "ivy")
        app_client.post("/app/memories", json={"text": "I like tea"}, headers=h)
        app_client.post("/app/reminders", json={"title": "call mum"}, headers=h)
        r = app_client.post("/app/reset", json={}, headers=h)
        assert r.json()["memories_removed"] == 1
        assert app_client.get("/app/memories", headers=h).json()["memories"] == []
        assert app_client.get("/app/reminders", headers=h).json() == []

    def test_reset_does_not_touch_others(self, app_client):
        a, b = login(app_client, "ivy"), login(app_client, "chris")
        app_client.post("/app/memories", json={"text": "ivy likes tea"}, headers=a)
        app_client.post("/app/memories", json={"text": "chris likes coffee"}, headers=b)
        app_client.post("/app/reset", json={}, headers=a)
        assert len(app_client.get("/app/memories", headers=b).json()["memories"]) == 1

    def test_reset_all_needs_api_key(self, app_client):
        h = login(app_client, "ivy")
        app_client.post("/app/memories", json={"text": "I like tea"}, headers=h)
        assert app_client.post("/app/reset-all", json={"api_key": "wrong"},
                               headers=h).status_code == 403
        r = app_client.post("/app/reset-all", json={"api_key": TEST_API_KEY},
                            headers=h)
        assert r.status_code == 200
        assert app_client.get("/app/memories", headers=h).json()["memories"] == []


class TestUiRoute:

    def test_ui_served(self, app_client):
        r = app_client.get("/ui")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    @pytest.mark.parametrize("path", [
        "/ui/../../.env", "/ui/..%2F..%2F.env", "/ui/%2e%2e/%2e%2e/.env.example",
    ])
    def test_ui_path_traversal_blocked(self, app_client, path):
        r = app_client.get(path)
        assert "MAEM_API_KEY" not in r.text
