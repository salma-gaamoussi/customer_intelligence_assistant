import ast
import re
from typing import Optional

from pydantic import BaseModel

MAX_QUESTION_LENGTH = 1000

INJECTION_REFUSAL_MESSAGE = (
    "I can't help with that request. Please ask a different question about customer data or policy documents."
)

# (pattern_name, compiled regex) — case-insensitive by construction (re.I).
INJECTION_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    (
        "ignore_instructions",
        re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions", re.I),
    ),
    (
        "disregard_rules",
        re.compile(r"(disregard|forget|override)\s+(the\s+|your\s+)?(rules|guidelines|instructions|prompt)", re.I),
    ),
    ("system_prompt", re.compile(r"system\s+prompt", re.I)),
    (
        "reveal_config",
        re.compile(
            r"(reveal|show|print|leak|display|(what|what's|tell me)\s+(is\s+)?)\s*"
            r"(your\s+|the\s+)?(config(uration)?|api\s*key|credentials?|env(ironment)?\s*variables?|secrets?)",
            re.I,
        ),
    ),
    (
        "roleplay_redirect",
        re.compile(
            r"(you are now|pretend (to be|you are)|new persona|jailbreak|"
            r"act as (if|though) you|from now on you are)",
            re.I,
        ),
    ),
]

# Defense-in-depth only — see module docstring. telco_readonly has no grants
# for any of these, so this list existing or not existing changes nothing
# about what the database will actually allow.
FORBIDDEN_SQL_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "CREATE",
)


class GuardrailResult(BaseModel):
    allowed: bool
    reason: Optional[str] = None
    guardrail_triggered: bool = False
    matched_pattern: Optional[str] = None


def check_prompt_injection(question: str) -> Optional[tuple[str, str]]:
    for name, pattern in INJECTION_PATTERNS:
        match = pattern.search(question)
        if match:
            return name, match.group(0)
    return None


def check_input(question: str) -> GuardrailResult:
    stripped = question.strip()

    if not stripped:
        return GuardrailResult(allowed=False, reason="Question cannot be empty.")

    if len(stripped) > MAX_QUESTION_LENGTH:
        return GuardrailResult(
            allowed=False,
            reason=f"Question is too long (max {MAX_QUESTION_LENGTH} characters).",
        )

    injection_match = check_prompt_injection(stripped)
    if injection_match is not None:
        pattern_name, matched_text = injection_match
        print(f"WARNING: guardrail flagged input — pattern={pattern_name!r} matched_text={matched_text!r}")
        return GuardrailResult(
            allowed=False,
            reason=INJECTION_REFUSAL_MESSAGE,
            guardrail_triggered=True,
            matched_pattern=pattern_name,
        )

    return GuardrailResult(allowed=True)


def check_sql_is_select_only(sql: str) -> GuardrailResult:
    normalized = sql.upper()
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized):
            print(f"WARNING: guardrail rejected SQL — matched_keyword={keyword!r} sql={sql!r}")
            return GuardrailResult(
                allowed=False,
                reason=f"Query rejected: contains forbidden keyword {keyword}.",
                guardrail_triggered=True,
                matched_pattern=keyword,
            )
    return GuardrailResult(allowed=True)


def enforce_row_limit(result_text: str, max_rows: int) -> str:
    if result_text.startswith("Error:"):
        return result_text

    try:
        rows = ast.literal_eval(result_text)
    except (ValueError, SyntaxError):
        return result_text

    if not isinstance(rows, list) or len(rows) <= max_rows:
        return result_text

    truncated = rows[:max_rows]
    return f"{truncated} ... [truncated to {max_rows} rows by guardrail]"
