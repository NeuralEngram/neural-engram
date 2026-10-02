# MAEM NeuralEngram

MAEM NeuralEngram is an AI assistant with human-like memory: it recalls what
users tell it, lets unused memories fade and strengthens repeated ones using
Ebbinghaus's forgetting curve, and alerts caregivers when Alzheimer's patients
start forgetting important things. An agentic, memory-augmented RAG system
built on each user's evolving memory.

## Features

- Chat assistant that remembers what you tell it and recalls it when relevant
- Memories fade over time (`R = e^(-t/S)`) and strengthen when recalled
- Three account types: standard user, Alzheimer's patient, caregiver
- Caregiver dashboard with alerts, memory health, reminders and trends
- Email alerts when a patient's important memories start to fade

## Tech stack

| Layer | Technology | Used for |
|---|---|---|
| Language | Python 3.10+ | Backend, agents, memory system |
| LLM | Anthropic Claude API (`claude-sonnet-5-5`) | Generating chat replies |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim) | Turning text into vectors for similarity search |
| Vector database | Qdrant (embedded, on disk) | Semantic memory: finding memories by meaning |
| Relational database | SQLite (WAL mode) | Episodic memory, accounts, reminders, alerts, trends |
| Numerical computing | NumPy | Vector similarity and decay calculations |
| Memory model | Ebbinghaus forgetting curve `R = e^(-t/S)` | Decay, reinforcement and forgetting |
| Backend framework | FastAPI + Uvicorn | REST API and serving the web app |
| Validation | Pydantic v2 | Request and response models |
| Authentication | PyJWT (HS256 tokens), PBKDF2-SHA256 password hashing | Login sessions and stored passwords |
| Security | API-key access gate, role-based access, CORS, SlowAPI rate limiting | Protecting data and routes |
| Configuration | python-dotenv | Loading settings from `.env` |
| Background jobs | Python `threading` scheduler | Decay, cleanup and alert checks every 5 minutes |
| Notifications | SMTP (Gmail), optional Twilio SMS | Caregiver alerts |
| Frontend | React 18 + Vite 5 | Web app (login, chat, memories, caregiver dashboard, settings) |
| Styling and charts | Plain CSS with light and dark themes, inline SVG charts | UI design and retention charts |
| Typography | Atkinson Hyperlegible | Readable text for elderly users |
| Architecture | Multi-agent pipeline: Planner → Executor → Memory | Intent routing, retrieval and response |
| Testing | pytest, httpx (FastAPI TestClient) | 1,056 automated tests |
| Evaluation | Custom harness + labelled retrieval benchmark | 29 quality gates; 93.3% retrieval accuracy |

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env
```

In `.env`, set at least:

```
ANTHROPIC_API_KEY=your-anthropic-key
MAEM_API_KEY=any-secret-access-key
```

For email alerts, also fill in the `SMTP_*` settings (see `.env.example`).

## Run

```bash
uvicorn api.main:app --reload
```

Open <http://localhost:8000/ui>, enter your `MAEM_API_KEY`, then sign in.

## Demo accounts

Password for each account is the username followed by `@123` (e.g. `will@123`).

| Type | Usernames |
|---|---|
| Standard | arjun, sid, grace, ivy, chris |
| Alzheimer's | will, henry, siri, zoya, riko |
| Caregiver | siya (Will, Henry, Siri), ansh (Zoya, Riko) |

## Tests and evaluation

```bash
pytest
python -m evaluation.evaluate
python -m evaluation.retrieval_eval
```

Retrieval accuracy on a 30-query labelled benchmark: **93.3%** (baseline 73.3%).

## Project structure

```
agents/         Planner and executor agents
memory/         Working, episodic and semantic memory, forgetting curve
orchestrator/   Connects agents, memory and the LLM
accounts/       Users, caregivers, reminders, alerts
api/            FastAPI backend
frontend/       React + Vite web app
evaluation/     Evaluation and retrieval benchmark
tests/          Automated tests
```

## License

MIT
