import json
import logging

from pydantic import ValidationError

from llm import async_call_llm
from models import FactualAssertion, Finding

logger = logging.getLogger(__name__)

_SYSTEM = """You are a fact-checker for legal documents. You verify factual assertions in a Motion for Summary Judgment against the primary source documents: a police report, medical records, and a witness statement.

For each factual assertion, check whether it is:
- Directly contradicted by text in the source documents
- Unsupported (no evidence for it in any source document)
- Missing important context that the source documents reveal

ONLY produce a finding when you detect a problem or important omission. If an assertion is fully supported by the source documents, omit it — do not return any entry for it.

When you do produce a finding, use this structure:
- finding_id: "f_fa_<assertion_id>" (e.g., "f_fa_fa_1")
- finding_type: EXACTLY one of these three strings (no other values are valid):
    "FACT_CONTRADICTION" — a source document directly contradicts the assertion
    "MISSING_EVIDENCE" — source documents reveal important context the MSJ omits
    "NOT_VERIFIABLE" — the assertion cannot be confirmed or contradicted from the provided documents
- severity: exactly one of: "critical", "high", "medium", "low"
- verdict: exactly one of: "CONTRADICTED", "UNSUPPORTED", "COULD_NOT_VERIFY"
  (never use "CONFIRMED" — supported assertions are simply omitted)
- subject_id: the assertion_id
- subject_summary: one-line description of the assertion being evaluated
- reasoning: 2-4 sentence explanation citing specific document text
- evidence: array of evidence objects. For EACH source document that is relevant, include one entry:
    "source_document": one of "police_report", "medical_records_excerpt", "witness_statement", "motion_for_summary_judgment"
    "excerpt": VERBATIM text from that document (exact copy, not paraphrase) — this is mandatory for any non-legal_knowledge source
    "relevance": one sentence explaining why this excerpt matters
- confidence: object with:
    "score": float 0.0–1.0
    "explanation": one sentence

CRITICAL RULE: For every finding of type FACT_CONTRADICTION or MISSING_EVIDENCE, the "excerpt" field in evidence MUST contain verbatim text from the source document. Do not paraphrase.

Return ONLY a JSON object with key "findings" containing an array of finding objects. No prose."""


async def agent_cross_document_checker(
    assertions: list[FactualAssertion],
    documents: dict[str, str],
) -> list[Finding]:
    """Check each MSJ factual assertion against the police report, medical records, and witness statement.

    All source documents are passed in a single context (~3k tokens combined). Each
    finding of type FACT_CONTRADICTION or MISSING_EVIDENCE must include verbatim
    excerpts from the source document as the primary defence against hallucination.

    Args:
        assertions: Factual assertions extracted by ``DocumentParser``.
        documents: All loaded case documents keyed by filename stem.

    Returns:
        Findings with types ``FACT_CONTRADICTION``, ``MISSING_EVIDENCE``, or
        ``NOT_VERIFIABLE``.

    Failure modes:
        - May hallucinate document text; mitigated by requiring verbatim excerpts.
        - Implicit contradictions requiring multi-step reasoning may be missed.
        - Per-item ``ValidationError``s are logged and the item is skipped.
    """
    logger.info("CrossDocumentChecker starting — %d assertions to check", len(assertions))
    assertions_payload = [a.model_dump() for a in assertions]
    docs_section = "\n\n".join(
        f"=== {name.upper().replace('_', ' ')} ===\n{text}"
        for name, text in documents.items()
        if name != "motion_for_summary_judgment"
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Check these factual assertions from the MSJ against the source documents.\n\n"
                f"ASSERTIONS:\n{json.dumps(assertions_payload, indent=2)}\n\n"
                f"SOURCE DOCUMENTS:\n{docs_section}"
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
            logger.warning("Skipping invalid cross-doc finding %s: %s", item.get("finding_id"), e)
    logger.info("CrossDocumentChecker complete — %d findings", len(findings))
    return findings
