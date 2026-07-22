# CLAUDE.md

## Project: Customer Intelligence Assistant
An AI assistant for a (fictional) telecom retention team. Agents ask plain-English
questions; the system routes each query to either a RAG pipeline over policy PDFs
or a SQL agent over the customer database, or refuses out-of-scope queries.

## Architecture
User → FastAPI /ask → input guardrail → LLM router (structured output: route,
confidence, reasoning) → one of:
- "sql"  → LangChain SQL agent → PostgreSQL `telco` schema
- "docs" → RAG chain → pgvector (`rag` schema) → answer with cited sources
- "out_of_scope" → polite refusal
Every request logged: query, route, confidence, latency, answer.
Offline: ingestion pipeline (PDF → chunk → embed → pgvector) and eval script.

## Stack — do not substitute
Python 3.11, FastAPI, LangChain, PostgreSQL 16 + pgvector (ONE instance, two
schemas: `telco`, `rag`), sentence-transformers (local embeddings), Docker Compose,
pytest. No LlamaIndex, no ChromaDB, no ORM beyond what LangChain needs.

## Structure
app/ (main.py, router.py, rag_chain.py, sql_agent.py, guardrails.py),
ingestion/ingest.py, evals/run_evals.py, data/ (telco.csv, documents/*.pdf)

## Key requirements
- Answers grounded in retrieved content only; system must say "I don't know"
  rather than invent (R4 — critical)
- Documents re-ingestible without rebuilds: idempotent ingestion
- Vendor-neutral: no hard cloud dependencies
- Config via .env (DATABASE_URL, EMBEDDING_MODEL, LLM keys), never hardcoded

## Conventions
- Type hints everywhere; Pydantic models for all structured LLM outputs
- Small functions, no clever one-liners — I need to explain every file in interviews
- After each component, tell me: key design decisions made and any trade-offs