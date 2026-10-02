"""
Manual persistence check for MAEM — confirms episodic (SQLite) and
semantic (Qdrant) memory both survive a full process restart, checked
independently, against a real running server.

Usage:
    1. Start the server normally, e.g.:
           uvicorn api.main:app
    2. In another terminal:
           python3 scripts/verify_persistence.py write
    3. Stop the server (Ctrl-C) and start it again:
           uvicorn api.main:app
    4. python3 scripts/verify_persistence.py read

If step 4 prints "BOTH TIERS SURVIVED THE RESTART", persistence is
working end to end. If it fails, the failure message tells you which
tier lost data.

Config (env vars, same ones the server itself reads):
    MAEM_API_BASE   default http://127.0.0.1:8000
    MAEM_API_KEY    required — same key the server is configured with
"""
import json
import os
import sys
import urllib.error
import urllib.request

API_BASE = os.getenv("MAEM_API_BASE", "http://127.0.0.1:8000")
API_KEY = os.getenv("MAEM_API_KEY", "")

AGENT_ID = "rajan"
CONTENT = "Rajan takes his heart medication at 8am every day"
EMBEDDING = [0.001 * i for i in range(384)]  # fixed, arbitrary — just needs to be consistent
MARKER_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".persistence_probe_memory_id"
)


def call(method, path, body=None):
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def write():
    if not API_KEY:
        sys.exit("MAEM_API_KEY is not set in this shell's environment.")

    status, result = call("POST", "/store", {
        "agent_id": AGENT_ID,
        "content": CONTENT,
        "embedding": EMBEDDING,
        "context": {},
    })
    if status != 200:
        sys.exit(f"POST /store failed ({status}): {result}")

    memory_id = result["memory_ids"]["memory_id"]
    with open(MARKER_FILE, "w") as f:
        f.write(memory_id)

    status, metrics = call("GET", f"/metrics/{AGENT_ID}")
    if status != 200 or metrics.get("total_memories", 0) < 1:
        sys.exit(f"Sanity check failed right after storing: {metrics}")

    print(f"Stored memory_id={memory_id} (agent={AGENT_ID}).")
    print(f"Now restart the server process, then run:\n  python3 {sys.argv[0]} read")


def read():
    if not API_KEY:
        sys.exit("MAEM_API_KEY is not set in this shell's environment.")
    if not os.path.exists(MARKER_FILE):
        sys.exit(f"No prior run found — run '{sys.argv[0]} write' first.")

    with open(MARKER_FILE) as f:
        memory_id = f.read().strip()

    ok = True

    # ── episodic tier, via /metrics ──
    status, metrics = call("GET", f"/metrics/{AGENT_ID}")
    if status != 200:
        print(f"FAIL: GET /metrics/{AGENT_ID} returned {status}: {metrics}")
        ok = False
    else:
        match = next((m for m in metrics.get("memories", []) if m["id"] == memory_id), None)
        if match is None:
            print(f"FAIL: episodic tier — memory_id {memory_id} not found after restart "
                  f"(total_memories={metrics.get('total_memories')})")
            ok = False
        else:
            print(f"OK: episodic tier survived the restart (retention={match['retention']}, "
                  f"stability_hours={match['stability_hours']}).")

    # ── semantic tier, via /retrieve (separate store — checked separately
    # on purpose, since these two tiers persist independently and either
    # one could fail without affecting the other) ──
    status, retrieved = call("POST", "/retrieve", {
        "agent_id": AGENT_ID,
        "query": CONTENT,
        "embedding": EMBEDDING,
        "top_k": 5,
    })
    if status != 200:
        print(f"FAIL: POST /retrieve returned {status}: {retrieved}")
        ok = False
    else:
        semantic_ids = [h.get("memory_id") for h in retrieved.get("semantic_memory", [])]
        if memory_id not in semantic_ids:
            print(f"FAIL: semantic tier — memory_id {memory_id} not in /retrieve results "
                  f"(got {semantic_ids})")
            ok = False
        else:
            print("OK: semantic tier survived the restart.")

    if ok:
        os.remove(MARKER_FILE)
        print("\nBOTH TIERS SURVIVED THE RESTART.")
    else:
        print("\nPersistence check FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("write", "read"):
        sys.exit(f"usage: python3 {sys.argv[0]} [write|read]")
    (write if sys.argv[1] == "write" else read)()
