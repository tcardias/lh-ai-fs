import asyncio
import logging
import time
from typing import Any

from agents.citation_verifier import agent_citation_verifier
from agents.cross_document_checker import agent_cross_document_checker
from agents.document_parser import agent_document_parser
from agents.judicial_memo_synthesizer import agent_judicial_memo_synthesizer
from agents.quote_verifier import agent_quote_verifier
from models import Finding, Severity, VerificationReport

logger = logging.getLogger(__name__)

_SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.CRITICAL: 0.20,
    Severity.HIGH: 0.10,
    Severity.MEDIUM: 0.05,
    Severity.LOW: 0.01,
}


def _compute_reliability_score(findings: list[Finding]) -> float:
    """Compute a reliability score for the MSJ based on finding severities.

    Applies a severity-weighted penalty to a baseline of 1.0:
    critical → −0.20, high → −0.10, medium → −0.05, low → −0.01.
    Result is clamped to [0, 1] and rounded to three decimal places.

    Args:
        findings: All findings produced by the verification agents.

    Returns:
        A float in [0.0, 1.0] where 1.0 means no issues detected.
    """
    penalty = sum(_SEVERITY_WEIGHTS.get(f.severity, 0.0) for f in findings)
    return max(0.0, round(1.0 - penalty, 3))


def _collect_results(*results: Any) -> tuple[list[Finding], list[str]]:
    """Separate successful agent outputs from exceptions.

    Each argument is either a ``list[Finding]`` (agent success) or an
    ``Exception`` captured by ``asyncio.gather(return_exceptions=True)``.

    Args:
        *results: Positional results from ``asyncio.gather``.

    Returns:
        A tuple of (combined findings list, human-readable error strings).
    """
    findings: list[Finding] = []
    errors: list[str] = []
    for result in results:
        if isinstance(result, Exception):
            errors.append(f"{type(result).__name__}: {result}")
            logger.error("Agent failure: %s", result, exc_info=result)
        elif isinstance(result, list):
            findings.extend(result)
    return findings, errors


async def run_pipeline(documents: dict[str, str]) -> VerificationReport:
    """Run the full BS Detector pipeline on the provided case documents.

    Executes in three steps:
      1. DocumentParser extracts citations, quotes, and assertions from the MSJ.
      2. CitationVerifier, QuoteVerifier, and CrossDocumentChecker run in parallel.
      3. JudicialMemoSynthesizer produces the judge-facing summary.

    Agent failures in step 2 are captured in ``pipeline_errors`` without aborting
    the pipeline. The report is always returned, possibly partial.

    Args:
        documents: Dict of filename stem → raw text. Must include
            ``"motion_for_summary_judgment"``.

    Returns:
        A fully populated ``VerificationReport``.

    Raises:
        KeyError: If ``"motion_for_summary_judgment"`` is absent from ``documents``.
    """
    start = time.perf_counter()
    logger.info("Pipeline started — %d documents loaded", len(documents))

    msj = documents["motion_for_summary_judgment"]

    parsed = await agent_document_parser(msj)
    logger.info(
        "Step 1 complete — %d citations, %d quotes, %d assertions",
        len(parsed.citations),
        len(parsed.quotes),
        len(parsed.factual_assertions),
    )

    citation_results, quote_results, consistency_results = await asyncio.gather(
        agent_citation_verifier(parsed.citations, msj),
        agent_quote_verifier(parsed.quotes, msj),
        agent_cross_document_checker(parsed.factual_assertions, documents),
        return_exceptions=True,
    )

    all_findings, pipeline_errors = _collect_results(citation_results, quote_results, consistency_results)
    logger.info(
        "Step 2 complete — %d findings, %d agent errors",
        len(all_findings),
        len(pipeline_errors),
    )

    memo = await agent_judicial_memo_synthesizer(all_findings)

    reliability = _compute_reliability_score(all_findings)
    report = VerificationReport(
        supporting_documents=[k for k in documents if k != "motion_for_summary_judgment"],
        parsed=parsed,
        findings=all_findings,
        judicial_memo=memo.text,
        top_findings=memo.top_findings,
        overall_reliability_score=reliability,
        pipeline_errors=pipeline_errors,
    )

    elapsed = time.perf_counter() - start
    logger.info(
        "Pipeline finished in %.1fs — %d findings, reliability score: %.3f",
        elapsed,
        len(all_findings),
        reliability,
    )
    return report
