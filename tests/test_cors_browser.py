# tests/test_cors_browser.py
"""
CORS enforced by a REAL browser, not a simulated header check.

Setup (all on loopback, ephemeral ports):
  - the real api/main.py app on uvicorn             http://127.0.0.1:<api>
  - an ALLOWED origin page (in ALLOWED_ORIGINS)     http://localhost:<ok>
  - a DISALLOWED origin page                        http://127.0.0.1:<evil>

Headless Chromium opens each origin page and runs, via page.evaluate,
the same fetch() you would paste into the DevTools console. Chrome's
own console errors are captured and asserted on.

Skipped automatically if Playwright/Chromium isn't installed. To run
locally:  pip install playwright && playwright install chromium
"""

import functools
import http.server
import importlib
import os
import socket
import socketserver
import sys
import threading
import time

import pytest

pytest.importorskip("playwright")
uvicorn = pytest.importorskip("uvicorn")

from tests.conftest import TEST_API_KEY, _ShellOrchestrator, hash_embed  # noqa: E402


def bound_socket():
    """
    A listening socket already bound to a free port. Binding port 0 and
    keeping the socket (instead of probing a port, closing it, and
    re-binding later) removes the window in which another process can
    take the port between the two steps.
    """
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(64)
    return s


def serve_blank_page(directory):
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=directory)
    handler.log_message = lambda *a: None
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


@pytest.fixture
def browser_env(real_memory, monkeypatch, tmp_path):
    import dotenv
    import orchestrator.orchestrator as orch_module

    (tmp_path / "index.html").write_text("<html><body>origin</body></html>")
    ok_srv, ok_port = serve_blank_page(str(tmp_path))
    evil_srv, evil_port = serve_blank_page(str(tmp_path))
    pages = [ok_srv, evil_srv]
    api_sock = bound_socket()
    api_port = api_sock.getsockname()[1]
    ok_origin = f"http://localhost:{ok_port}"
    evil_origin = f"http://127.0.0.1:{evil_port}"

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("MAEM_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("ALLOWED_ORIGINS", ok_origin)
    monkeypatch.setattr(orch_module, "Orchestrator",
                        lambda: _ShellOrchestrator(real_memory.store))
    sys.modules.pop("api.main", None)
    api = importlib.import_module("api.main")

    server = uvicorn.Server(uvicorn.Config(api.app, log_level="warning"))
    t = threading.Thread(target=server.run,
                         kwargs={"sockets": [api_sock]}, daemon=True)
    t.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"

    from playwright.sync_api import sync_playwright
    if os.path.isdir("/opt/pw-browsers"):
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch()
    except Exception as e:  # noqa: BLE001
        pw.stop()
        server.should_exit = True
        for p in pages:
            p.shutdown()
            p.server_close()
        pytest.skip(f"Chromium not available: {e}")

    yield {
        "api": f"http://127.0.0.1:{api_port}",
        "ok": ok_origin, "evil": evil_origin,
        "browser": browser, "memory": real_memory,
    }

    browser.close()
    pw.stop()
    for p in pages:
        p.shutdown()
        p.server_close()
    server.should_exit = True
    t.join(timeout=10)
    api_sock.close()
    sys.modules.pop("api.main", None)


def console_fetch(env, origin, js_body):
    """Open `origin` in a fresh page, run JS, return (result, console)."""
    page = env["browser"].new_page()
    console = []
    page.on("console", lambda m: console.append(m.text))
    page.goto(origin + "/index.html")
    result = page.evaluate(js_body, {"api": env["api"], "key": TEST_API_KEY})
    page.wait_for_timeout(200)  # let Chrome flush its CORS console error
    page.close()
    return result, console


READ_JS = """
async ({api, key}) => {
  try {
    const r = await fetch(api + "/metrics/rajan",
                          {headers: {"X-API-Key": key}});
    return "OK:" + r.status + ":" + JSON.stringify(await r.json());
  } catch (e) { return "BLOCKED:" + e.message; }
}
"""

WRITE_JS = """
async ({api, key}) => {
  const emb = Array(384).fill(0); emb[0] = 1;
  try {
    const r = await fetch(api + "/store", {
      method: "POST",
      headers: {"X-API-Key": key, "Content-Type": "application/json"},
      body: JSON.stringify({agent_id: "rajan", content: "written by page",
                            embedding: emb})});
    return "OK:" + r.status;
  } catch (e) { return "BLOCKED:" + e.message; }
}
"""

NO_CORS_JS = """
async ({api}) => {
  // What a malicious page CAN send without a preflight: no custom
  // headers allowed, so no X-API-Key.
  await fetch(api + "/store", {method: "POST", mode: "no-cors",
    headers: {"Content-Type": "text/plain"},
    body: '{"agent_id":"rajan","content":"csrf","embedding":[]}'});
  return "sent";
}
"""


class TestCorsInRealBrowser:

    def test_allowed_origin_can_read(self, browser_env):
        browser_env["memory"].store.store(
            "rajan", "rajan likes tea", hash_embed("rajan likes tea"))

        result, _ = console_fetch(browser_env, browser_env["ok"], READ_JS)

        assert result.startswith("OK:200:")
        assert "rajan likes tea" in result

    def test_disallowed_origin_read_blocked_by_browser(self, browser_env):
        browser_env["memory"].store.store(
            "rajan", "rajan likes tea", hash_embed("rajan likes tea"))

        result, console = console_fetch(
            browser_env, browser_env["evil"], READ_JS)

        assert result.startswith("BLOCKED:")
        assert "rajan likes tea" not in result
        assert any("blocked by CORS policy" in line for line in console), \
            console

    def test_allowed_origin_can_write(self, browser_env):
        result, _ = console_fetch(browser_env, browser_env["ok"], WRITE_JS)

        assert result == "OK:200"
        contents = [e.content for e in
                    browser_env["memory"].episodic.get_all()]
        assert contents == ["written by page"]

    def test_disallowed_origin_write_never_reaches_server(self, browser_env):
        """
        The custom X-API-Key header forces a preflight; the preflight is
        refused, so the browser never sends the POST at all — the write
        is prevented, not merely hidden from the page.
        """
        result, console = console_fetch(
            browser_env, browser_env["evil"], WRITE_JS)

        assert result.startswith("BLOCKED:")
        assert browser_env["memory"].episodic.get_all() == []
        assert any("preflight" in line for line in console), console

    def test_no_cors_request_cannot_carry_the_key(self, browser_env):
        result, _ = console_fetch(
            browser_env, browser_env["evil"], NO_CORS_JS)

        assert result == "sent"
        time.sleep(0.2)
        assert browser_env["memory"].episodic.get_all() == []
