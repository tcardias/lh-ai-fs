import json
import logging

from llm import async_call_llm
from models import Finding, JudicialMemo

logger = logging.getLogger(__name__)

_SYSTEM = """You are a judicial clerk drafting a memo for a judge reviewing a Motion for Summary Judgment.

You will receive a list of findings produced by a multi-agent verification pipeline. Your task is to:

1. Select the 3–5 most significant findings by a combination of severity and confidence score.
   Prioritize findings that are both high-severity AND high-confidence.
   A critical finding with 0.5 confidence ranks lower than a high finding with 0.9 confidence.

2. Write a single paragraph (4–6 sentences) addressed to the judge that:
   - States the overall reliability of the brief concisely
   - Names the most serious issues found (date discrepancy, PPE claim, quote accuracy, etc.)
   - Identifies which source documents contradict the MSJ's claims
   - Notes the practical implication for the motion

3. Return top_findings as an ordered list (most significant first) with:
   - finding_id: the finding's ID
   - one_line_summary: concise description suitable for a bullet-point list
   - severity: the finding's severity level
   - confidence_score: the confidence score (0.0–1.0)

Return ONLY a JSON object with keys "text" (the paragraph) and "top_findings" (the ordered list). No prose outside the JSON."""


async def agent_judicial_memo_synthesizer(findings: list[Finding]) -> JudicialMemo:
    """Synthesize all pipeline findings into a judge-facing memo.

    Selects the 3–5 most significant findings by severity and confidence,
    writes a one-paragraph memo, and returns an ordered list of top findings.

    Args:
        findings: All findings from CitationVerifier, QuoteVerifier, and
            CrossDocumentChecker.

    Returns:
        A ``JudicialMemo`` with the memo text and ranked ``top_findings``.

    Failure modes:
        - May over-weight legally interesting issues vs. clear factual contradictions.
        - May lose specific evidence references when synthesising across many findings.
    """
    logger.info("JudicialMemoSynthesizer starting — %d findings to synthesize", len(findings))
    findings_payload = [f.model_dump() for f in findings]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Synthesize these {len(findings)} findings into a judicial memo.\n\n"
                f"FINDINGS:\n{json.dumps(findings_payload, indent=2)}"
            ),
        },
    ]
    raw = await async_call_llm(messages, response_format={"type": "json_object"})
    result = JudicialMemo.model_validate(json.loads(raw))
    logger.info(
        "JudicialMemoSynthesizer complete — %d top findings selected",
        len(result.top_findings),
    )
    return result
