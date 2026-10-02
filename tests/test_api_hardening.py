# tests/test_api_hardening.py
"""
Phase 2 hardening: API-key auth, CORS, rate limiting, fail-fast on a
missing ANTHROPIC_API_KEY.

Uses the real api/main.py app (conftest.api_app): real auth dependency,
real CORSMiddleware, real slowapi limiter, real memory behind it.
"""

import os
import re
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ALLOWED_TEST_ORIGIN, TEST_API_KEY, hash_embed


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATE_LIMIT = 30  # "30/minute" on /chat and /store in api/main.py


@pytest.fixture
def client(api_app):
    return TestClient(api_app.app)


def store_body(content="alice grows tomatoes", agent_id="alice"):
    return {
        "agent_id": agent_id,
        "content": content,
        "embedding": hash_embed(content),
    }


# ─── API key ───────────────────────────────────────────────

class TestApiKey:

    PROTECTED = [
        ("get", "/metrics/alice", None),
        ("get", "/memories/alice", None),
        ("get", "/context/alice", None),
        ("get", "/alerts/alice", None),
        ("get", "/scheduler/status", None),
        ("post", "/store", store_body()),
        ("post", "/chat", {"message": "hi", "user_id": "alice"}),
        ("post", "/decay", None),
        ("post", "/cleanup", None),
        ("post", "/feedback/alice/x", {"positive": True}),
    ]

    @pytest.mark.parametrize("method, path, body", PROTECTED)
    def test_missing_key_rejected(self, client, method, path, body):
        resp = getattr(client, method)(path, json=body) if body \
            else getattr(client, method)(path)
        assert resp.status_code == 401

    @pytest.mark.parametrize("method, path, body", PROTECTED)
    def test_wrong_key_rejected(self, client, method, path, body):
        headers = {"X-API-Key": TEST_API_KEY + "-nope"}
        resp = getattr(client, method)(path, json=body, headers=headers) \
            if body else getattr(client, method)(path, headers=headers)
        assert resp.status_code == 401

    def test_valid_key_accepted(self, client, auth_headers):
        assert client.get(
            "/metrics/alice", headers=auth_headers
        ).status_code == 200

    def test_valid_key_store_really_stores(
        self, client, auth_headers, real_memory
    ):
        resp = client.post("/store", json=store_body(), headers=auth_headers)

        assert resp.status_code == 200
        mid = resp.json()["memory_ids"]["memory_id"]
        assert real_memory.episodic.get_by_id(mid).agent_id == "alice"

    def test_rejected_request_has_no_side_effect(self, client, real_memory):
        client.post("/store", json=store_body())
        assert real_memory.episodic.get_all() == []

    def test_health_check_is_public(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_unset_server_key_fails_closed(
        self, client, monkeypatch, auth_headers
    ):
        monkeypatch.delenv("MAEM_API_KEY")
        resp = client.get("/metrics/alice", headers=auth_headers)
        assert resp.status_code == 500

    def test_every_non_public_route_is_protected(self, api_app, client):
        """
        Guard against a future route being added on `app` instead of
        `router` (which would silently skip auth): every operation in
        the OpenAPI schema except the public ones must 401 without a key.
        """
        public = {("get", "/"), ("get", "/ui")}
        checked = 0
        for path, ops in api_app.app.openapi()["paths"].items():
            concrete = re.sub(r"\{[^}]+\}", "x", path)
            for method in ops:
                if (method, path) in public:
                    continue
                resp = client.request(method.upper(), concrete, json={})
                assert resp.status_code == 401, f"{method.upper()} {path}"
                checked += 1
        assert checked >= 15


# ─── CORS ──────────────────────────────────────────────────

class TestCors:

    def preflight(self, client, origin):
        return client.options(
            "/metrics/alice",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )

    def test_disallowed_origin_preflight_rejected(self, client):
        resp = self.preflight(client, "https://evil.example.com")
        assert resp.status_code == 400
        assert "access-control-allow-origin" not in resp.headers

    def test_disallowed_origin_gets_no_allow_header(self, client, auth_headers):
        resp = client.get(
            "/metrics/alice",
            headers={**auth_headers, "Origin": "https://evil.example.com"},
        )
        assert "access-control-allow-origin" not in resp.headers

    def test_allowed_origin_preflight_ok(self, client):
        resp = self.preflight(client, ALLOWED_TEST_ORIGIN)
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == \
            ALLOWED_TEST_ORIGIN

    def test_allowed_origin_on_real_request(self, client, auth_headers):
        resp = client.get(
            "/metrics/alice",
            headers={**auth_headers, "Origin": ALLOWED_TEST_ORIGIN},
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == \
            ALLOWED_TEST_ORIGIN

    def test_no_wildcard(self, api_app):
        assert "*" not in api_app.allowed_origins

    def test_origins_come_from_env(self, api_app):
        assert api_app.allowed_origins == [ALLOWED_TEST_ORIGIN]

    def test_default_is_localhost_only(self, api_app, monkeypatch):
        import importlib
        import sys

        monkeypatch.delenv("ALLOWED_ORIGINS")
        sys.modules.pop("api.main", None)
        fresh = importlib.import_module("api.main")

        assert fresh.allowed_origins
        for origin in fresh.allowed_origins:
            assert "localhost" in origin or "127.0.0.1" in origin


# ─── Rate limiting ─────────────────────────────────────────

class TestRateLimit:

    def test_store_limit_triggers_on_request_n_plus_one(
        self, client, auth_headers
    ):
        codes = [
            client.post(
                "/store", json=store_body(f"note number {i}"),
                headers=auth_headers,
            ).status_code
            for i in range(RATE_LIMIT + 1)
        ]
        assert codes[:RATE_LIMIT] == [200] * RATE_LIMIT
        assert codes[RATE_LIMIT] == 429

    def test_chat_limit_triggers_on_request_n_plus_one(
        self, client, auth_headers
    ):
        codes = [
            client.post(
                "/chat",
                json={"message": f"note number {i}", "user_id": "alice"},
                headers=auth_headers,
            ).status_code
            for i in range(RATE_LIMIT + 1)
        ]
        assert codes[:RATE_LIMIT] == [200] * RATE_LIMIT
        assert codes[RATE_LIMIT] == 429

    def test_rate_limited_request_does_not_store(
        self, client, auth_headers, real_memory
    ):
        for i in range(RATE_LIMIT):
            client.post("/store", json=store_body(f"note {i}"),
                        headers=auth_headers)
        before = len(real_memory.episodic.get_all())

        resp = client.post("/store", json=store_body("one too many"),
                           headers=auth_headers)

        assert resp.status_code == 429
        assert len(real_memory.episodic.get_all()) == before

    def test_unlimited_route_not_throttled(self, client, auth_headers):
        for _ in range(RATE_LIMIT + 10):
            assert client.get(
                "/metrics/alice", headers=auth_headers
            ).status_code == 200

    def test_unauthenticated_calls_do_not_consume_budget(
        self, client, auth_headers
    ):
        """
        Characterisation: auth runs before the limiter, so bad-key
        requests are rejected (401) without counting toward the limit.
        """
        for _ in range(RATE_LIMIT + 5):
            assert client.post(
                "/store", json=store_body()
            ).status_code == 401

        assert client.post(
            "/store", json=store_body(), headers=auth_headers
        ).status_code == 200


# ─── Fail fast on missing ANTHROPIC_API_KEY ────────────────

class TestStartupFailsFast:

    @pytest.fixture
    def orch_module(self, monkeypatch):
        import orchestrator.orchestrator as orch_module

        # If validation ever moved below these, the test would notice
        # heavyweight setup happening before the key check.
        monkeypatch.setattr(orch_module, "SentenceTransformer", MagicMock())
        monkeypatch.setattr(orch_module, "MemoryScheduler", MagicMock())
        monkeypatch.setattr(orch_module, "EpisodicMemoryStore", MagicMock())
        return orch_module

    @pytest.mark.parametrize(
        "value",
        [None, "", "   ", "your-anthropic-api-key", "changeme",
         "YOUR-KEY-HERE"],
    )
    def test_missing_or_placeholder_key_raises(
        self, orch_module, monkeypatch, value
    ):
        if value is None:
            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        else:
            monkeypatch.setenv("ANTHROPIC_API_KEY", value)

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            orch_module.Orchestrator()

    def test_nothing_heavy_constructed_before_failing(
        self, orch_module, monkeypatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(RuntimeError):
            orch_module.Orchestrator()

        orch_module.SentenceTransformer.assert_not_called()
        orch_module.EpisodicMemoryStore.assert_not_called()
        orch_module.MemoryScheduler.assert_not_called()

    def test_real_looking_key_passes_validation(self, orch_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-realish")
        orch_module._validate_anthropic_api_key()  # no raise

    def test_server_import_exits_cleanly_without_key(self, tmp_path):
        """
        End-to-end: importing api.main (what uvicorn does) in a fresh
        process with no key must exit non-zero with a clear message —
        and before touching any data files.
        """
        env = dict(os.environ)
        env["ANTHROPIC_API_KEY"] = ""  # "" also blocks .env override
        env["MAEM_DB_PATH"] = str(tmp_path / "should_not_exist.db")
        env["QDRANT_STORAGE_PATH"] = str(tmp_path / "qdrant_should_not_exist")
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (PROJECT_ROOT, env.get("PYTHONPATH")) if p
        )

        proc = subprocess.run(
            [sys.executable, "-c", "import api.main"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert proc.returncode != 0
        assert "RuntimeError" in proc.stderr
        assert "ANTHROPIC_API_KEY is not set" in proc.stderr
        assert not (tmp_path / "should_not_exist.db").exists()
        assert not (tmp_path / "qdrant_should_not_exist").exists()
