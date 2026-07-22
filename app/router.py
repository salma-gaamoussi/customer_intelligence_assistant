from typing import Literal, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

CONFIDENCE_THRESHOLD = 0.7

FALLBACK_MESSAGE = (
    "I'm not confident enough to answer that automatically. Could you rephrase your "
    "question - for example, specify whether you're asking about a specific customer's "
    "account/billing/service details, or about a retention team policy or procedure?"
)

ROUTER_SYSTEM_PROMPT = """You are a router for a telecom retention-team assistant. Classify \
each question into exactly one route: "sql", "docs", or "out_of_scope".

Route "sql" — questions answerable by querying the telco.customers database table. Columns:
customer_id, gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines,
internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv,
streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges,
churn. This is per-customer account, billing, and service data: counts, averages, filters,
lookups on individual customers or customer segments.
Examples:
- "How many customers are on a month-to-month contract?"
- "What's customer 7590-VHVEG's monthly charge?"
- "List customers who churned and have no tech support."

Route "docs" — questions answerable from internal policy documents covering: retention discount
authorization tiers (how much agents/supervisors may discount, by tenure and contract type), the
former-customer win-back procedure (eligibility windows and exclusions for re-engaging
disconnected customers), the customer issue escalation procedure (severity levels, response and
resolution targets, escalation path), and the new-customer 90-day onboarding guide (onboarding
timeline, first-bill-shock prevention). This also covers metadata about these documents (who owns
a policy, its document ID) and any question that plausibly falls in the retention/customer-policy
workplace domain, even if you're not sure the specific document covers it. Route those to "docs"
and let retrieval decide — do not preemptively classify a plausible policy question as
out_of_scope just because it doesn't obviously match one of the four topics above; retrieval will
correctly report back if the documents don't actually cover it.
Examples:
- "How much discount can an agent offer without supervisor approval?"
- "Can we win back a customer who disconnected 200 days ago?"
- "What's the first response time target for a Severity 2 issue?"
- "Which customers are required to receive a 30-day wellness call?" (onboarding-guide content,
  phrased around "customers" but it's asking about a policy rule, not database rows)
- "Who owns policy document RET-POL-021?" (metadata about a policy document)
- "What are the discount rules for business accounts?" (plausible policy question — route to
  docs even if this specific carve-out may not be documented; let retrieval find out and report
  back rather than assuming out_of_scope up front)
- "How much annual leave do retention agents get?" (workplace/HR-flavored but framed as a policy
  question about retention-team staff — route to docs rather than assuming it's unrelated)

Route "out_of_scope" — anything else: general knowledge, small talk, requests unrelated to
telecom customer data or these specific policies, or requests to take actions this assistant
can't perform (e.g. modifying data, sending emails, making purchases).
Examples:
- "What's the capital of France?"
- "Write me a poem."
- "Delete customer 1234's account."

Give your genuine confidence for the classification — do not default to a high number just
because you produced an answer. If the question is ambiguous between routes, could plausibly
belong to more than one, or you're not sure it's answerable at all, reflect that with a lower
confidence score rather than guessing. This includes vague questions that are plausibly in scope
but missing key details (e.g. "tell me about the account" without naming a customer, or "what's
the policy" without saying which one) — these are not confidently out_of_scope just because they
lack detail; score them low-confidence instead of picking a route to commit to. This also
includes questions that sound answerable but are missing a parameter the answer actually depends
on — e.g. the retention discount tiers vary by both customer tenure and contract type, so a
question asking for "the" discount percentage without giving both is not something you can
confidently answer with a single number; score it low-confidence rather than picking one tier
from the table and presenting it as the answer.
Examples:
- "Tell me about the account" (no customer specified) -> low confidence, plausibly "sql"
- "What's the policy?" (no topic specified) -> low confidence, plausibly "docs"
- "What percentage discount can I offer to an at-risk customer?" (discount tiers depend on
  tenure and contract type, neither given) -> low confidence, do not confidently answer "docs\""""


class RouteDecision(BaseModel):
    route: Literal["sql", "docs", "out_of_scope"] = Field(
        description="Which subsystem should handle this question."
    )
    confidence: float = Field(description="Confidence in this route, from 0.0 to 1.0.")
    reasoning: str = Field(description="One or two sentences explaining the decision.")


class RouterOutcome(BaseModel):
    route: Literal["sql", "docs", "out_of_scope", "clarify"]
    confidence: float
    reasoning: str
    fallback_message: Optional[str] = None


def get_router_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def classify_question(question: str) -> RouteDecision:
    llm = get_router_llm()
    structured_llm = llm.with_structured_output(RouteDecision)
    return structured_llm.invoke(
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    )


def route_question(question: str) -> RouterOutcome:
    decision = classify_question(question)
    if decision.confidence < CONFIDENCE_THRESHOLD:
        return RouterOutcome(
            route="clarify",
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            fallback_message=FALLBACK_MESSAGE,
        )
    return RouterOutcome(
        route=decision.route,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        fallback_message=None,
    )
