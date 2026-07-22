"""Offline eval harness: runs evals/eval_dataset.json straight through the app's
components (no HTTP) and reports router accuracy, retrieval hit rate, and answer
quality (substring + LLM-as-judge faithfulness).

Usage:
    python -m evals.run_evals --strategy fixed
    python -m evals.run_evals --strategy recursive
"""
import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.main import OUT_OF_SCOPE_MESSAGE
from app.rag_chain import RetrievedChunk, ask_rag_chain
from app.router import route_question
from app.sql_agent import ask_sql_agent

load_dotenv()

DATASET_PATH = Path("evals/eval_dataset.json")
RESULTS_DIR = Path("evals/results")
JUDGE_MODEL = "gpt-4o-mini"

FAITHFULNESS_PROMPT = """You are a strict fact-checker reviewing a RAG system's answer.

Question: {question}

Retrieved source chunks:
{chunks_block}

Answer to check:
{answer}

Is every claim in the answer supported by the retrieved chunks above? An answer that
says it doesn't have the information, and makes no other factual claims, counts as
faithful (YES) — there is nothing unsupported in it. Respond with your judgment and a
one-line reason."""


class EvalCase(BaseModel):
    id: str
    question: str
    expected_route: str
    expected_answer_contains: list[str] = []
    expected_answer_not_contains: list[str] = []
    expected_source: Optional[str] = None
    notes: str = ""


class FaithfulnessJudgment(BaseModel):
    faithful: Literal["YES", "NO"]
    reason: str


class CaseResult(BaseModel):
    id: str
    question: str
    expected_route: str
    predicted_route: str
    confidence: float
    router_correct: bool
    answer: str
    expected_source: Optional[str] = None
    retrieved_sources: list[str] = []
    retrieval_hit: Optional[bool] = None
    substring_pass: Optional[bool] = None
    missing_substrings: list[str] = []
    not_contains_pass: Optional[bool] = None
    violating_substrings: list[str] = []
    faithfulness: Optional[str] = None
    faithfulness_reason: Optional[str] = None
    error: Optional[str] = None


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(**case) for case in data["cases"]]


def score_retrieval_hit(case: EvalCase, predicted_route: str, retrieved_sources: list[str]) -> Optional[bool]:
    if case.expected_route != "docs" or case.expected_source is None:
        return None
    if predicted_route != "docs":
        return False
    return case.expected_source in retrieved_sources


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(",", "")).strip().lower()


def score_substrings(expected: list[str], answer: str) -> tuple[Optional[bool], list[str]]:
    if not expected:
        return None, []
    answer_norm = _normalize_for_match(answer)
    missing = [s for s in expected if _normalize_for_match(s) not in answer_norm]
    return len(missing) == 0, missing


def score_not_contains(expected_not: list[str], answer: str) -> tuple[Optional[bool], list[str]]:
    if not expected_not:
        return None, []
    answer_norm = _normalize_for_match(answer)
    violating = [s for s in expected_not if _normalize_for_match(s) in answer_norm]
    return len(violating) == 0, violating


def get_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0)


def score_faithfulness(
    case: EvalCase, predicted_route: str, answer: str, chunks: list[RetrievedChunk]
) -> tuple[Optional[str], Optional[str]]:
    if case.expected_route != "docs" or predicted_route != "docs" or not chunks:
        return None, None

    chunks_block = "\n\n".join(f'Source: "{c.source_file}"\n{c.chunk_text}' for c in chunks)
    prompt = FAITHFULNESS_PROMPT.format(question=case.question, chunks_block=chunks_block, answer=answer)

    llm = get_judge_llm()
    judgment = llm.with_structured_output(FaithfulnessJudgment).invoke(prompt)
    return judgment.faithful, judgment.reason


def run_case(case: EvalCase) -> CaseResult:
    """Route the question exactly as app/main.py would, then score every metric
    group against whatever the live system actually did, including a routing
    miss, which correctly fails retrieval hit rate and answer-quality checks
    downstream rather than being silently skipped
    """
    try:
        outcome = route_question(case.question)
        predicted_route = outcome.route

        answer = ""
        retrieved_chunks: list[RetrievedChunk] = []

        if predicted_route == "sql":
            result = ask_sql_agent(case.question)
            answer = result.answer
        elif predicted_route == "docs":
            rag_result = ask_rag_chain(case.question)
            answer = rag_result.answer
            retrieved_chunks = rag_result.chunks
        elif predicted_route == "clarify":
            answer = outcome.clarification_message or ""
        else:
            answer = OUT_OF_SCOPE_MESSAGE

        retrieved_sources = sorted({c.source_file for c in retrieved_chunks})
        retrieval_hit = score_retrieval_hit(case, predicted_route, retrieved_sources)
        substring_pass, missing = score_substrings(case.expected_answer_contains, answer)
        not_contains_pass, violating = score_not_contains(case.expected_answer_not_contains, answer)
        faithfulness, faithfulness_reason = score_faithfulness(case, predicted_route, answer, retrieved_chunks)

        return CaseResult(
            id=case.id,
            question=case.question,
            expected_route=case.expected_route,
            predicted_route=predicted_route,
            confidence=outcome.confidence,
            router_correct=predicted_route == case.expected_route,
            answer=answer,
            expected_source=case.expected_source,
            retrieved_sources=retrieved_sources,
            retrieval_hit=retrieval_hit,
            substring_pass=substring_pass,
            missing_substrings=missing,
            not_contains_pass=not_contains_pass,
            violating_substrings=violating,
            faithfulness=faithfulness,
            faithfulness_reason=faithfulness_reason,
        )
    except Exception as exc:
        return CaseResult(
            id=case.id,
            question=case.question,
            expected_route=case.expected_route,
            predicted_route="error",
            confidence=0.0,
            router_correct=False,
            answer="",
            expected_source=case.expected_source,
            error=str(exc),
        )


def _format_bool(value: Optional[bool]) -> str:
    if value is None:
        return "  -  "
    return " PASS" if value else " FAIL"


def print_case_table(results: list[CaseResult]) -> None:
    header = f"{'id':<16}{'route(exp/pred)':<22}{'conf':>6}  {'router':<6}{'retr':<6}{'subst':<6}{'faith':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        route_col = f"{r.expected_route}/{r.predicted_route}"
        router_col = " OK" if r.router_correct else " NO"
        faith_col = r.faithfulness if r.faithfulness else ("  -  " if r.faithfulness is None else r.faithfulness)
        print(
            f"{r.id:<16}{route_col:<22}{r.confidence:>6.2f}  "
            f"{router_col:<6}{_format_bool(r.retrieval_hit):<6}{_format_bool(r.substring_pass):<6}{faith_col:<6}"
        )
        if r.error:
            print(f"    ERROR: {r.error}")


def compute_summary(results: list[CaseResult]) -> dict:
    total = len(results)
    router_correct = sum(1 for r in results if r.router_correct)

    confusion: dict[str, dict[str, int]] = {}
    for r in results:
        confusion.setdefault(r.expected_route, {}).setdefault(r.predicted_route, 0)
        confusion[r.expected_route][r.predicted_route] += 1

    retrieval_cases = [r for r in results if r.retrieval_hit is not None]
    retrieval_hits = sum(1 for r in retrieval_cases if r.retrieval_hit)

    substring_cases = [r for r in results if r.substring_pass is not None]
    substring_passes = sum(1 for r in substring_cases if r.substring_pass)

    not_contains_cases = [r for r in results if r.not_contains_pass is not None]
    not_contains_passes = sum(1 for r in not_contains_cases if r.not_contains_pass)

    faithfulness_cases = [r for r in results if r.faithfulness is not None]
    faithfulness_passes = sum(1 for r in faithfulness_cases if r.faithfulness == "YES")

    return {
        "total_cases": total,
        "router_accuracy": router_correct / total if total else 0.0,
        "router_confusion": confusion,
        "retrieval_hit_rate": retrieval_hits / len(retrieval_cases) if retrieval_cases else None,
        "retrieval_cases_evaluated": len(retrieval_cases),
        "substring_pass_rate": substring_passes / len(substring_cases) if substring_cases else None,
        "substring_cases_evaluated": len(substring_cases),
        "not_contains_pass_rate": not_contains_passes / len(not_contains_cases) if not_contains_cases else None,
        "faithfulness_pass_rate": faithfulness_passes / len(faithfulness_cases) if faithfulness_cases else None,
        "faithfulness_cases_evaluated": len(faithfulness_cases),
    }


def print_summary(summary: dict) -> None:
    print("\n=== Summary ===")
    print(f"Cases: {summary['total_cases']}")
    print(f"Router accuracy: {summary['router_accuracy']:.0%}")

    print("Router confusion (rows=expected, cols=predicted):")
    routes = sorted({r for row in summary["router_confusion"].values() for r in row} | set(summary["router_confusion"]))
    header = "  " + "".join(f"{r:>14}" for r in routes)
    print(header)
    for expected in routes:
        row = summary["router_confusion"].get(expected, {})
        print(f"  {expected:<12}" + "".join(f"{row.get(r, 0):>14}" for r in routes))

    def fmt_rate(key: str, count_key: str) -> str:
        rate = summary[key]
        return "n/a" if rate is None else f"{rate:.0%} ({summary[count_key]} cases)"

    print(f"Retrieval hit rate: {fmt_rate('retrieval_hit_rate', 'retrieval_cases_evaluated')}")
    print(f"Substring pass rate: {fmt_rate('substring_pass_rate', 'substring_cases_evaluated')}")
    print(f"Faithfulness pass rate: {fmt_rate('faithfulness_pass_rate', 'faithfulness_cases_evaluated')}")


def save_results(strategy: str, timestamp: str, results: list[CaseResult], summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{strategy}_{timestamp}.json"
    payload = {
        "strategy": strategy,
        "timestamp": timestamp,
        "summary": summary,
        "cases": [r.model_dump() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline eval suite.")
    parser.add_argument("--strategy", choices=["fixed", "recursive"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["RAG_CHUNKING_STRATEGY"] = args.strategy

    cases = load_cases(DATASET_PATH)
    results = [run_case(case) for case in cases]

    print_case_table(results)
    summary = compute_summary(results)
    print_summary(summary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = save_results(args.strategy, timestamp, results, summary)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
