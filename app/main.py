import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from app.guardrails import check_input
from app.rag_chain import ask_rag_chain
from app.router import route_question
from app.sql_agent import ask_sql_agent

load_dotenv()

LOG_FILE = Path("logs/requests.jsonl")
OUT_OF_SCOPE_MESSAGE = "I'm sorry, that's outside what I can help with."

app = FastAPI(title="Customer Intelligence Assistant")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    route: str
    confidence: float
    sources: list[str]
    latency_ms: float


def log_request(entry: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def dispatch(route: str, question: str) -> tuple[str, list[str]]:
    if route == "sql":
        result = ask_sql_agent(question)
        return result.answer, result.sql_queries

    if route == "docs":
        result = ask_rag_chain(question)
        sources = sorted({chunk.source_file for chunk in result.chunks})
        return result.answer, sources

    return OUT_OF_SCOPE_MESSAGE, []


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    start = time.perf_counter()

    guardrail = check_input(request.question)
    if not guardrail.allowed:
        response = AskResponse(
            answer=guardrail.reason or "That question isn't allowed.",
            route="blocked",
            confidence=1.0,
            sources=[],
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        log_request(_log_entry(request.question, response, guardrail_triggered=guardrail.guardrail_triggered))
        return response

    outcome = route_question(request.question)

    if outcome.route == "clarify":
        answer = outcome.clarification_message or "Could you clarify your question?"
        sources: list[str] = []
        route_taken = "clarify"
    else:
        answer, sources = dispatch(outcome.route, request.question)
        route_taken = outcome.route

    response = AskResponse(
        answer=answer,
        route=route_taken,
        confidence=outcome.confidence,
        sources=sources,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
    log_request(_log_entry(request.question, response))
    return response


def _log_entry(query: str, response: AskResponse, guardrail_triggered: bool = False) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "route": response.route,
        "confidence": response.confidence,
        "latency_ms": response.latency_ms,
        "answer": response.answer,
        "guardrail_triggered": guardrail_triggered,
    }
