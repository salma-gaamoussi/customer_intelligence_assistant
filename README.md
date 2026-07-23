# Customer Intelligence Assistant

An AI assistant for a (fictional) telecom retention team. Agents ask plain-English
questions and the system figures out which subsystem can actually answer them — a
SQL agent over the customer database, a RAG pipeline over internal policy PDFs, or
a polite refusal if the question's out of scope.

## Why this exists

Retention agents end up needing two pretty different kinds of answers in the same
conversation: "how many customers churned on month-to-month contracts" is a
database question, "how much discount can I authorize for a 2-year customer" is a
policy question. I wanted to build the routing between those two properly — LLM
router with real confidence scores, answers grounded in retrieved content only, and
an offline eval harness so I'm not just eyeballing whether it "seems fine."

## Architecture

```
User → FastAPI /ask → input guardrail → LLM router (route, confidence, reasoning)
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                          ▼                         ▼
              route="sql"                route="docs"            route="out_of_scope"
          LangChain SQL agent          RAG chain (pgvector)          polite refusal
        → PostgreSQL `telco` schema  → `rag` schema, cited sources
```

If the router isn't confident, it doesn't guess — it short-circuits into a
`clarify` response asking for more detail. Every request (query, route, confidence,
latency, answer) gets logged to `logs/requests.jsonl`.

One PostgreSQL 16 instance (`pgvector/pgvector:pg16`) holds two schemas:
- `telco` — the customer table, queried through a **read-only** Postgres role
  (`telco_readonly`) so the SQL agent literally cannot write, no matter what the LLM
  generates.
- `rag` — chunked, embedded policy document text (local `sentence-transformers`
  embeddings, no embedding API involved).

## Stack

Python 3.11 · FastAPI · LangChain · PostgreSQL 16 + pgvector · sentence-transformers
(local embeddings) · Docker Compose · pytest

## Project structure

```
app/
  main.py         FastAPI app: /ask endpoint, request logging, dispatch to route
  router.py       LLM router — structured output (route, confidence, reasoning)
  sql_agent.py    LangChain SQL agent + guardrail-wrapped query tool
  rag_chain.py    Embed question → pgvector similarity search → grounded answer
  guardrails.py   Input (prompt injection, length) and output (SQL, row-count) checks
ingestion/
  ingest.py       PDF → chunk (fixed or recursive) → embed → pgvector, idempotent
  load_telco.py   Load data/Telco_Customer_Churn.csv into telco.customers
evals/
  run_evals.py    Offline harness: router accuracy, retrieval hit rate, faithfulness
  eval_dataset.json
data/
  Telco_Customer_Churn.csv
  documents/      Policy PDFs (discounts, win-back, escalation, onboarding)
db/init.sql       Schemas, pgvector extension, read-only role + grants
docker-compose.yml
```

## Setup

**1. Start Postgres + pgvector**
```
docker compose up -d
```

**2. Create a virtualenv and install dependencies**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure environment**

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` (used by the router, SQL
agent, and RAG generation step). Everything else has a working local default.

**4. Load the customer data**
```
python -m ingestion.load_telco
```

**5. Ingest the policy PDFs**
```
python -m ingestion.ingest --strategy recursive
```
Safe to re-run — each file gets deleted and re-inserted per
`(source_file, chunking_strategy)`, so nothing duplicates.

**6. Run the API**
```
uvicorn app.main:app --reload
```

## Usage

```
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How much discount can an agent offer without supervisor approval?"}'
```

You get back the answer, the route taken, the router's confidence, cited sources
(for `docs`) or executed queries (for `sql`), and latency.

## Evaluation

```
python -m evals.run_evals --strategy recursive
```

This scores every case against what the pipeline actually did, not against the
"expected" route — so a misrouted `docs` question also fails retrieval and answer
quality for that case, instead of getting excluded as not-applicable. Otherwise a
routing bug can hide behind a retrieval number that looks fine in isolation.

Where things stand right now (both chunking strategies): **100% router accuracy**,
**86% retrieval hit rate** — the one remaining miss is an intentional trap case
where two policies are genuinely easy to conflate. Recursive chunking has a small
but consistent edge on answer-quality phrasing (81% vs 75% substring pass rate)
even though retrieval performance between the two strategies is identical.

## Design decisions worth knowing about

**Clarify instead of guessing.** The router has to give a genuine confidence score
rather than defaulting high just because it produced *an* answer. Anything under
0.7 gets a clarifying question back instead of a possibly-wrong route.

**Defense-in-depth on SQL, not one clever check.** The read-only Postgres role
means writes are impossible at the database layer no matter what the guardrail code
does. On top of that there are two independent guardrail layers — one caps the
query before execution, one caps the *returned row count* after. I added the
second one after realizing the first only rejects a query with no `LIMIT` at all;
a query the agent wrote with its own `LIMIT 999999` would've sailed straight
through.

**Idempotent ingestion.** Each PDF gets deleted then re-inserted, keyed on
`(source_file, chunking_strategy)`, before its new chunks go in. Re-ingesting a
changed document never duplicates rows and doesn't need a rebuild.

**Cascade scoring in evals.** A routing miss fails every downstream metric for that
case rather than getting marked N/A. It's the more honest number — I didn't want a
router bug hiding inside an artificially good-looking retrieval or faithfulness
score.

**The router iteration was error-analysis-driven, not prompt-guessing.** Baseline
was 76% router accuracy with 5 concrete misroutes. I turned each miss into a
few-shot example paired with a generalizable rule — not just the bare Q&A — e.g.
"route plausible policy questions to docs even if you're not sure the specific
document covers it; let retrieval decide." One pass got router accuracy to 100%,
and that alone pulled retrieval hit rate up from 57% to 86% as a side effect —
nothing changed in `rag_chain.py` or `sql_agent.py`, the router was just starving
those cases before.

**Guardrail patterns got tuned against real phrasing, not just read for
correctness.** Testing turned up both a false negative ("What is your API key?" —
the original pattern only caught imperative phrasing, not questions) and a false
positive ("Can you act as a translator for this text?" — flagged as a jailbreak
attempt by a pattern that was too broad).

## Limitations

- The prompt-injection and SQL-safety checks are regex heuristics, not a formally
  verified guardrail. The real safety boundary for SQL is the read-only DB role —
  the guardrail is defense-in-depth on top of that, not the main line of defense.
- The eval dataset is small and hand-written, so the accuracy numbers describe this
  dataset, not a general claim about router quality.
- Single-instance Postgres with no auth beyond the compose defaults — this is a
  local dev setup, not something to point at the internet as-is.
