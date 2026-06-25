import json
import logging

from pydantic import ValidationError

from llm import async_call_llm
from models import Finding, Quote

logger = logging.getLogger(__name__)

_SYSTEM = """You are a legal quote verifier. Your job is to check whether direct quotes in a legal brief accurately reproduce the attributed source.

For each quote provided, assess whether the quoted text:
- Accurately represents what the attributed case or source actually says
- Contains inserted words, removed words, or altered punctuation that changes meaning
- Is taken out of context in a way that reverses or distorts the original meaning
- Appears to be fabricated (attributed to a real case that never said it)

ONLY produce a finding when you detect a problem. If a quote appears accurate, omit it — do not return any entry for it.

When you do produce a finding, use this structure:
- finding_id: "f_q_<quote_id>" (e.g., "f_q_q_1")
- finding_type: EXACTLY one of these two strings (no other values are valid):
    "QUOTE_MISMATCH" — the quote is inaccurate, altered, fabricated, or taken out of context
    "NOT_VERIFIABLE" — you cannot confirm or deny the quote from available knowledge
- severity: exactly one of: "critical", "high", "medium", "low"
  Use "critical" when the alteration reverses the legal meaning; "high" when it significantly expands the holding
- verdict: exactly one of: "ALTERED", "FABRICATED", "COULD_NOT_VERIFY"
  (never use "CONFIRMED" — confirmed quotes are simply omitted)
- subject_id: the quote_id
- subject_summary: one line naming the source and what is claimed
- reasoning: identify the SPECIFIC word or phrase that appears wrong
- evidence: array of objects, each with:
    "source_document": "legal_knowledge"
    "excerpt": ""
    "relevance": one sentence
- confidence: object with:
    "score": float 0.0–1.0
    "explanation": one sentence

IMPORTANT: The Privette v. Superior Court actual holding is that a hirer is "presumptively not liable" — a rebuttable presumption, NOT an absolute immunity. Any quote claiming a hirer is "never liable" must be flagged as QUOTE_MISMATCH with verdict FABRICATED or ALTERED.

Return ONLY a JSON object with key "findings" containing an array of finding objects. No prose."""


async def agent_quote_verifier(quotes: list[Quote], msj_text: str) -> list[Finding]:
    """Verify each direct quote for accuracy against the attributed source.

    Checks for inserted words, removed words, changed meaning, or fabricated quotes.
    Requires the model to identify the specific word or phrase that is wrong.
    Only returns findings for problematic quotes; confirmed ones are omitted.

    Args:
        quotes: Extracted quotes from ``DocumentParser``.
        msj_text: Full MSJ text provided as context.

    Returns:
        Findings with types ``QUOTE_MISMATCH`` or ``NOT_VERIFIABLE``.

    Failure modes:
        - Cannot verify obscure case quotes without source access.
        - Paraphrased quotes that are semantically close may be missed.
        - Per-item ``ValidationError``s are logged and the item is skipped.
    """
    logger.info("QuoteVerifier starting — %d quotes to verify", len(quotes))
    quotes_payload = [q.model_dump() for q in quotes]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Verify these direct quotes from the Motion for Summary Judgment.\n\n"
                f"QUOTES:\n{json.dumps(quotes_payload, indent=2)}\n\n"
                f"MSJ CONTEXT (for reference):\n{msj_text}"
            ),
        },
    ]
    raw = await async_call_llm(messages, response_format={"type": "json_object"})
    data = json.loads(raw)
    findings: list[Finding] = []
    for item in data["findings"]:
        try:
            findings.append(Finding.model_validate(item))
        except ValidationError as e:
            logger.warning("Skipping invalid quote finding %s: %s", item.get("finding_id"), e)
    logger.info("QuoteVerifier complete — %d findings", len(findings))
    return findings
