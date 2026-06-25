import json
import logging

from pydantic import ValidationError

from llm import async_call_llm
from models import Citation, Finding

logger = logging.getLogger(__name__)

_SYSTEM = """You are a legal citation verifier with expertise in California and federal case law.

For each citation provided, assess:
1. Whether the case plausibly exists (name, reporter, year, court are consistent)
2. Whether the stated proposition accurately represents what the case actually holds
3. Whether the pin page is consistent with the proposition

ONLY produce a finding when you detect a problem. If a citation appears accurate, omit it — do not return any entry for it.

When you do produce a finding, use this structure:
- finding_id: "f_cit_<citation_id>" (e.g., "f_cit_cit_1")
- finding_type: EXACTLY one of these four strings (no other values are valid):
    "UNSUPPORTED_CITATION" — the case exists but the proposition overstates or misrepresents the holding
    "POSSIBLY_FABRICATED_CASE" — the citation details are internally inconsistent or the case is unrecognizable
    "LEGAL_OVERSTATEMENT" — the case exists and is relevant, but the MSJ inflates its scope
    "NOT_VERIFIABLE" — you cannot confirm or deny from available knowledge
- severity: exactly one of: "critical", "high", "medium", "low"
- verdict: exactly one of: "CONTRADICTED", "UNSUPPORTED", "FABRICATED", "ALTERED", "COULD_NOT_VERIFY"
  (never use "CONFIRMED" here — confirmed citations are simply omitted)
- subject_id: the citation_id
- subject_summary: one-line description of the citation
- reasoning: your detailed explanation (2-4 sentences)
- evidence: array of objects, each with:
    "source_document": "legal_knowledge"
    "excerpt": ""
    "relevance": one sentence
- confidence: object with:
    "score": float 0.0–1.0
    "explanation": one sentence

IMPORTANT: If you cannot verify a citation, use NOT_VERIFIABLE — do NOT invent knowledge.
The Privette v. Superior Court case (5 Cal.4th 689) is a real California Supreme Court case from 1993.
Seabright Insurance Co. v. US Airways, Inc. (52 Cal.4th 590) is a real California Supreme Court case from 2011.

Return ONLY a JSON object with key "findings" containing an array of finding objects. No prose."""


async def agent_citation_verifier(citations: list[Citation], msj_text: str) -> list[Finding]:
    """Verify each legal citation against model knowledge of case law.

    Assesses whether each case plausibly exists, whether the stated proposition
    accurately represents the holding, and whether the pin page is consistent.
    Only returns findings for problematic citations; confirmed ones are omitted.

    Args:
        citations: Extracted citations from ``DocumentParser``.
        msj_text: Full MSJ text provided as context.

    Returns:
        Findings with types ``UNSUPPORTED_CITATION``, ``POSSIBLY_FABRICATED_CASE``,
        ``LEGAL_OVERSTATEMENT``, or ``NOT_VERIFIABLE``. Empty list if all pass.

    Failure modes:
        - May fabricate knowledge about unrecognized citations despite instructions.
        - Subtle misquotations in well-known cases may be missed without Westlaw access.
        - Per-item ``ValidationError``s are logged and the item is skipped.
    """
    logger.info("CitationVerifier starting — %d citations to verify", len(citations))
    citations_payload = [c.model_dump() for c in citations]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Verify these citations from the Motion for Summary Judgment.\n\n"
                f"CITATIONS:\n{json.dumps(citations_payload, indent=2)}\n\n"
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
            logger.warning("Skipping invalid citation finding %s: %s", item.get("finding_id"), e)
    logger.info("CitationVerifier complete — %d findings", len(findings))
    return findings
