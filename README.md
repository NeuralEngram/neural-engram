# 🧠 MAEM NeuralEngram — Multi-Agent Ebbinghaus Memory

A multi-agent AI assistant with biologically-inspired memory: what it learns
fades over time following **Ebbinghaus's Forgetting Curve**, strengthens when
it is recalled again, and — for monitored patients — alerts a caregiver by
email when an important memory is about to be forgotten.

---

## 📌 Overview

Most chatbots either forget everything when the session ends or remember
everything forever. NeuralEngram sits in between, like human memory:

- Every message is stored with a **stability** value and a **retention**
  score that decays over time: `R = e^(-t / S)`.
- Recalling or repeating a memory **reinforces** it — stability grows, so it
  decays more slowly next time.
- Memories whose retention falls below a threshold are **forgotten**
  (pruned) by a background scheduler.
- Important memories (medicine, doctor, family, appointments…) that start
  fading trigger a **caregiver alert** — shown on the dashboard and sent by
  email.

The primary use case is an **Alzheimer's patient companion**: a patient
profile (`rajan`) decays 3× faster than normal, so the system can
demonstrate memory loss and caregiver intervention in minutes instead of
days.

---

## 🏗️ Architecture

```
                    ┌──────────────────────────┐
  Browser / API ──► │  FastAPI  (api/main.py)  │  auth · CORS · rate-limit
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │       Orchestrator       │
                    └────────────┬─────────────┘
                                 ▼
      PlannerAgent ──► ExecutorAgent ──► Claude API (response)
            │               │
            └──────┬────────┘
                   ▼
            MemoryStore / MemoryOrchestrator
        ┌──────────┼──────────────┐
        ▼          ▼              ▼
   Working     Episodic        Semantic
   memory      (SQLite)        (Qdrant vectors)
   (last 20)   retention,      similarity
               stability       search
                   │
                   ▼
   MemoryScheduler (every 5 min) ──► decay · prune · caregiver alerts ──► Email
```

- **PlannerAgent** detects intent, decides whether and how much to
  retrieve, does the turn's single retrieval, and reinforces the memories
  that were used.
- **ExecutorAgent** builds the prompt from the retrieved memories and
  recent conversation, calls the LLM, and stores the turn.
- **MemoryAgent / MemoryStore** store, retrieve, reinforce and forget
  memories across the three tiers.
- **MemoryScheduler** runs decay and cleanup in the background and triggers
  caregiver notifications.

---

## 🔎 Retrieval pipeline (RAG)

NeuralEngram is an **agentic, memory-augmented RAG system**: instead of a
fixed document set, it retrieves from the user's own long-term memory,
which grows, fades and strengthens over time.

One chat turn:

1. **Intent routing** (`PlannerAgent`) — the message is classified as
   *memory*, *reasoning*, *general* or *small talk*. Personal references
   ("my", "did I", "where do I") count as memory questions.
   - Small talk ("hi", "ok thanks") → **no retrieval at all**.
   - Otherwise top-k is set by intent: memory 8, reasoning 6, general 4.
2. **One embedding, one retrieval** — the query is embedded once and
   retrieved once; the result and the embedding are handed to
   `ExecutorAgent`, which reuses both.
3. **Hybrid search** — semantic (Qdrant vector index) + episodic (cached
   unit vectors, scored with one matrix–vector product; stored memories
   are never re-embedded at query time).
4. **Relevance cut-off** — only memories with cosine similarity ≥
   `RETRIEVAL_MIN_SCORE` survive. Forgotten-but-not-yet-pruned memories
   and the assistant's own replies are excluded.
5. **Fusion & ranking** — cross-tier dedup, ranked by relevance (ties
   broken by retention), packed into a 4 000-token budget (an oversized
   memory is skipped, not allowed to block the rest).
6. **Generate** — Claude answers with the memory context.
7. **Reinforce** — only after a successful answer, and only the
   memories that were actually used, get reinforced (stability ↑).
   Unrelated memories keep fading, as the forgetting curve intends.

**Measuring it.** `python -m evaluation.retrieval_eval` runs a labelled
benchmark (22 answerable questions with gold memories + 8 no-answer
queries) and writes `evaluation/RETRIEVAL_RESULTS.md` with hit rate,
precision, recall, F1, MRR, noise rejection, irrelevant memories per
query and latency — baseline vs the current pipeline, plus a sweep to
pick the best `RETRIEVAL_MIN_SCORE` for your embedding model.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM | Anthropic Claude API |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector DB | Qdrant (embedded, on-disk) |
| Episodic DB | SQLite (WAL mode) |
| API | FastAPI + Uvicorn, Pydantic v2, SlowAPI |
| Notifications | SMTP email (Gmail), optional Twilio SMS |
| Frontend | React 18 + Vite (served by FastAPI at `/ui`) |
| Accounts | SQLite, PBKDF2 password hashing, JWT sessions |
| Testing | pytest, httpx |

---

## 📂 Project Structure

| Path | Description |
|---|---|
| `agents/` | `PlannerAgent`, `ExecutorAgent`, `MemoryAgent`, shared `BaseAgent` |
| `memory/ebbinghaus.py` | Forgetting-curve maths: retention, reinforcement, thresholds |
| `memory/working_memory.py` | Short-term buffer of recent turns |
| `memory/episodic_memory.py` | SQLite-backed episodic memory with decay |
| `memory/semantic_memory.py` | Qdrant-backed vector memory |
| `memory/memory_orchestrator.py` | Coordinates the three memory tiers |
| `memory/memory_store.py` | Public memory API used by the agents |
| `memory/retrieval_config.py` | Retrieval tuning (relevance cut-off, top-k per intent) |
| `memory/scheduler.py` | Background decay / cleanup loop |
| `memory/notifier.py` | Caregiver alert detection + email/SMS sending |
| `orchestrator/orchestrator.py` | Wires everything together; LLM calls |
| `api/main.py` | REST API, serves the web app at `/ui` |
| `api/app_routes.py` | Application API (`/app/...`): login, roles, reminders, alerts |
| `accounts/` | Users, caregiver links, reminders, alert history, trends |
| `frontend/` | React + Vite web app (`src/components/` one file per screen area) |
| `api/maem_multiagent.html` | Older single-file dashboard (used only if `frontend/dist` is missing) |
| `evaluation/evaluate.py` | Evaluation harness (writes `evaluation/RESULTS.md`) |
| `evaluation/retrieval_eval.py` | Retrieval benchmark (writes `evaluation/RETRIEVAL_RESULTS.md`) |
| `scripts/` | Terminal chat, demo, and persistence-check scripts |
| `tests/` | Unit and integration tests |
| `docs/` | Architecture notes |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or newer
- An Anthropic API key
- Node.js 18+ (only to change the frontend; a built copy is included)
- (Optional) A Gmail account with an App Password, for email alerts

### 1. Install

```bash
git clone https://github.com/<your-username>/maem_neuralengram.git
cd maem_neuralengram

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB).

### 2. Configure

```bash
# Windows:
copy .env.example .env
# macOS / Linux:
cp .env.example .env
```

Edit `.env`:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Your Anthropic API key |
| `MAEM_API_KEY` | ✅ | Any secret string. It is the access key that unlocks the web app, and API clients send it as the `X-API-Key` header |
| `MAEM_JWT_SECRET` | — | Signs login sessions (default: derived from `MAEM_API_KEY`) |
| `MAEM_TOKEN_HOURS` | — | How long a login lasts (default `12`) |
| `MAEM_APP_DB_PATH` | — | Accounts database (default `maem_app.db` next to the memory DB) |
| `CLAUDE_MODEL` | — | Chat model (default `claude-sonnet-5-5`) |
| `RETRIEVAL_MIN_SCORE` | — | Relevance cut-off for retrieval, 0–1 (default `0.30`) |
| `RETRIEVE_ASSISTANT_REPLIES` | — | `true` to let the bot's own replies be retrieved (default `false`) |
| `ALLOWED_ORIGINS` | — | CORS origins (default: localhost) |
| `NOTIFY_AGENT_IDS` | — | Extra API-only users to monitor (Alzheimer's accounts are monitored automatically) |
| `AGENT_DECAY_MULTIPLIERS` | — | Faster decay per user, e.g. `rajan:3.0` |
| `QDRANT_STORAGE_PATH` | — | Vector store folder (default `./qdrant_data`) |
| `MAEM_DB_PATH` | — | SQLite file (default `./multi_agent_memory.db`) |
| `ENABLE_EMAIL_ALERTS` | — | `true` to send caregiver emails |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` | if email on | SMTP login (Gmail: `smtp.gmail.com`, `587`, your address, a 16-char App Password) |
| `ALERT_EMAIL_FROM`, `ALERT_EMAIL_TO` | if email on | Sender and caregiver addresses |

> `.env` is read only at startup — restart the server after editing it.

### 3. Run

```bash
uvicorn api.main:app --reload
```
Open <http://localhost:8000/ui>, enter your `MAEM_API_KEY` to unlock the
app, then sign in. Interactive API docs: <http://localhost:8000/docs>.

**Changing the frontend** (needs Node.js):
```bash
cd frontend
npm install          # first time only
npm run dev          # live-reload at http://localhost:5173/ui/ (API proxied to :8000)
npm run build        # writes frontend/dist, which /ui serves
```

**Terminal chat / scripted demo**
```bash
python scripts/run_system.py
python scripts/run_demo.py
```

---

## 🖥️ The web app

### Accounts and roles

| Account type | What they get |
|---|---|
| **Standard** | Personal memory tracking: chat, memories, reminders. Never linked to a caregiver. |
| **Alzheimer's support** | Everything above, plus a "today" card, gentle reminders of fading important memories, larger text, a severity-based decay speed, and one caregiver. |
| **Caregiver** | A dashboard for their own Alzheimer's patients only: fading-memory alerts (mark as handled, history), memory health, retention trend, reminders and memories they can add, email alert settings. |

Access is layered: the **access key** (`MAEM_API_KEY`) unlocks the app,
then each person **signs in** with their username and password. A user sees
only their own data; a caregiver sees only their linked patients.

### Accounts created on first run

| Username | Type | Notes |
|---|---|---|
| `arjun`, `sid`, `grace`, `ivy`, `chris` | Standard | |
| `will` | Alzheimer's, mild | forgets 1.5× faster; caregiver Siya |
| `henry` | Alzheimer's, mild | 2× faster; caregiver Siya |
| `siri` | Alzheimer's, moderate | 3× faster; caregiver Siya |
| `zoya` | Alzheimer's, moderate | 4× faster; caregiver Ansh |
| `riko` | Alzheimer's, severe | 5× faster; caregiver Ansh |
| `siya`, `ansh` | Caregiver | |

Default password: the username followed by `@123` (for example `will@123`).
Change it in **Settings → Account**.

### Screens

- **Home** (users): chat, quick questions, "Remember this" to save an
  important fact, today's reminders and relevant memories, and reminders
  set by you or your caregiver.
- **Memories** (users): every memory with retention, importance, priority,
  times recalled and stability. Text fades as retention drops; filter by
  strong / fading / at risk, sort, refresh a memory, add one.
- **Dashboard** (caregivers): pick a patient; alerts, memory health,
  retention trend, reminders and memories.
- **Settings**: account and password, light/dark theme, email alerts
  (caregivers), memory tools (simulate time, compare forgetting), and
  start fresh (delete your memories, or every account's with the access key).

### Caregiver alerts
An alert fires for an Alzheimer's account when a memory that contains a
critical keyword (*medicine, medication, doctor, appointment, emergency,
hospital, blood pressure, insulin, tablet, daughter, son, family, home*)
drops below **50 % retention** (warning) or **30 %** (critical). Checks run
every 5 minutes and right after a time simulation. The caregiver sees it on
their dashboard and, if they turned on email alerts in Settings (and SMTP is
configured in `.env`), gets one email per alert level.

---

## 🔌 Main API Endpoints

All routes except `GET /` and `GET /ui` require the `X-API-Key` header.
The web app uses the `/app/...` routes, which also need a login token
(`Authorization: Bearer …` from `POST /app/auth/login`); see `/docs`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/ui` | Dashboard |
| `POST` | `/chat` | Send a message (`{"message", "user_id"}`) |
| `GET` | `/metrics/{agent_id}` | Retention / stability per memory |
| `GET` | `/memories/{agent_id}` | List memories |
| `POST` | `/reinforce` | Reinforce a memory |
| `POST` | `/feedback/{agent_id}/{memory_id}` | Positive/negative feedback |
| `POST` | `/simulate_time` | Age a user's memories (demo) |
| `GET` | `/alerts/{agent_id}` | Current caregiver alerts |
| `POST` | `/alerts/{agent_id}/notify` | Send pending alerts now; shows whether email is enabled |
| `POST` | `/decay`, `/cleanup` | Run decay / prune manually |
| `GET` | `/scheduler/status` | Background scheduler status |
| `POST` | `/evaluate` | Memory vs no-memory comparison (optional answer accuracy) |

See <http://localhost:8000/docs> for request/response schemas.

---

## 🧪 Testing & Evaluation

```bash
pytest                                 # full test suite
python -m evaluation.evaluate          # correctness gates → evaluation/RESULTS.md
python -m evaluation.retrieval_eval    # retrieval benchmark → evaluation/RETRIEVAL_RESULTS.md
```

To check that memory survives a server restart, follow the steps in
`scripts/verify_persistence.py`.

The evaluation measures:
- Retention accuracy against the forgetting curve
- Caregiver-alert precision (right memories, right level, no duplicates)
- Feedback / reinforcement effects on stability
- Retrieval quality and efficiency (precision, recall, MRR, noise
  rejection, re-embeddings per query)

The dashboard's **Eval** button calls `POST /evaluate`, which compares
answers with and without memory. Pass `expected_answers` (one per
question, alternatives separated by `|`) to also score answer accuracy.

---

## 🧬 The Forgetting Curve

```
R = e^(-t / S)
```

- **R** — retention (0–1)
- **t** — hours since the memory was last reviewed
- **S** — stability in hours; grows with each reinforcement and with
  importance, and shrinks for users with a decay multiplier > 1

Each reinforcement increases **S**, flattening the curve so the memory lasts
longer — mimicking how spaced repetition strengthens human memory.

---

## 🗺️ Roadmap

- [ ] Configurable decay profiles from the dashboard
- [ ] External Qdrant server support for multi-process deployments
- [ ] Multi-modal memory (image, audio)
- [ ] Caregiver dashboard with alert history

---

## 📄 License

MIT License.

## 🙏 Acknowledgements

Inspired by Hermann Ebbinghaus's research on human memory and forgetting,
adapted for multi-agent AI systems.
