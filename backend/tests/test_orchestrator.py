"""Tests for the orchestrator: pipeline flow, partial failures, and scoring."""

from unittest.mock import AsyncMock, patch

import pytest

from models import (
    Citation,
    Confidence,
    Evidence,
    FactualAssertion,
    Finding,
    FindingType,
    JudicialMemo,
    ParsedDocument,
    Severity,
    TopFinding,
    Verdict,
)
from orchestrator import _collect_results, _compute_reliability_score, run_pipeline

# ── Fixtures ───────────────────────────────────────────────────────────────────


def make_finding(finding_id: str = "f_1", severity: str = "high") -> Finding:
    return Finding(
        finding_id=finding_id,
        finding_type=FindingType.FACT_CONTRADICTION,
        severity=Severity(severity),
        verdict=Verdict.CONTRADICTED,
        subject_id="fa_1",
        subject_summary="Test",
        reasoning="Contradiction found.",
        evidence=[Evidence(source_document="police_report", excerpt="text", relevance="reason")],
        confidence=Confidence(score=0.9, explanation="High certainty"),
    )


EMPTY_PARSED = ParsedDocument(citations=[], quotes=[], factual_assertions=[])
EMPTY_MEMO = JudicialMemo(text="No issues found.", top_findings=[])

SAMPLE_DOCS = {
    "motion_for_summary_judgment": "MSJ text here.",
    "police_report": "Police report text.",
}


# ── _compute_reliability_score ─────────────────────────────────────────────────


def test_reliability_no_findings():
    assert _compute_reliability_score([]) == 1.0


def test_reliability_one_critical():
    assert _compute_reliability_score([make_finding(severity="critical")]) == pytest.approx(0.8)


def test_reliability_one_high():
    assert _compute_reliability_score([make_finding(severity="high")]) == pytest.approx(0.9)


def test_reliability_one_medium():
    assert _compute_reliability_score([make_finding(severity="medium")]) == pytest.approx(0.95)


def test_reliability_clamped_at_zero():
    findings = [make_finding(severity="critical")] * 10
    assert _compute_reliability_score(findings) == 0.0


def test_reliability_mixed_severities():
    findings = [
        make_finding("f_1", "critical"),
        make_finding("f_2", "high"),
        make_finding("f_3", "low"),
    ]
    # 1.0 - 0.20 - 0.10 - 0.01 = 0.69
    assert _compute_reliability_score(findings) == pytest.approx(0.69)


# ── _collect_results ───────────────────────────────────────────────────────────


def test_collect_all_success():
    f1, f2 = make_finding("f_1"), make_finding("f_2")
    findings, errors = _collect_results([f1], [f2], [])
    assert len(findings) == 2
    assert errors == []


def test_collect_with_exception():
    f1 = make_finding("f_1")
    findings, errors = _collect_results([f1], ValueError("agent down"), [])
    assert len(findings) == 1
    assert len(errors) == 1
    assert "ValueError" in errors[0]
    assert "agent down" in errors[0]


def test_collect_all_exceptions():
    _, errors = _collect_results(RuntimeError("a"), ValueError("b"))
    assert len(errors) == 2


def test_collect_empty():
    findings, errors = _collect_results([], [], [])
    assert findings == []
    assert errors == []


# ── run_pipeline (mocked agents) ──────────────────────────────────────────────


async def test_pipeline_happy_path():
    finding = make_finding()
    parsed = ParsedDocument(
        citations=[
            Citation(
                citation_id="cit_1",
                case_name="Test v. Case",
                reporter="Cal.4th",
                year=2000,
                proposition="Test prop",
                msj_section="III",
            )
        ],
        quotes=[],
        factual_assertions=[
            FactualAssertion(
                assertion_id="fa_1",
                msj_section="II",
                text="Some fact.",
                assertion_type="other",
            )
        ],
    )
    memo = JudicialMemo(
        text="The brief has issues.",
        top_findings=[
            TopFinding(
                finding_id="f_1",
                one_line_summary="Issue",
                severity=Severity.HIGH,
                confidence_score=0.9,
            )
        ],
    )

    with (
        patch("orchestrator.agent_document_parser", new=AsyncMock(return_value=parsed)),
        patch(
            "orchestrator.agent_citation_verifier",
            new=AsyncMock(return_value=[finding]),
        ),
        patch("orchestrator.agent_quote_verifier", new=AsyncMock(return_value=[])),
        patch("orchestrator.agent_cross_document_checker", new=AsyncMock(return_value=[])),
        patch(
            "orchestrator.agent_judicial_memo_synthesizer",
            new=AsyncMock(return_value=memo),
        ),
    ):
        report = await run_pipeline(SAMPLE_DOCS)

    assert report.case_id == "BC-2023-04851"
    assert len(report.findings) == 1
    assert report.pipeline_errors == []
    assert report.judicial_memo == "The brief has issues."
    assert report.overall_reliability_score == pytest.approx(0.9)
    assert report.supporting_documents == ["police_report"]


async def test_pipeline_agent_failure_captured():
    memo = EMPTY_MEMO

    with (
        patch(
            "orchestrator.agent_document_parser",
            new=AsyncMock(return_value=EMPTY_PARSED),
        ),
        patch(
            "orchestrator.agent_citation_verifier",
            new=AsyncMock(side_effect=RuntimeError("timeout")),
        ),
        patch("orchestrator.agent_quote_verifier", new=AsyncMock(return_value=[])),
        patch("orchestrator.agent_cross_document_checker", new=AsyncMock(return_value=[])),
        patch(
            "orchestrator.agent_judicial_memo_synthesizer",
            new=AsyncMock(return_value=memo),
        ),
    ):
        report = await run_pipeline(SAMPLE_DOCS)

    assert len(report.pipeline_errors) == 1
    assert "RuntimeError" in report.pipeline_errors[0]
    assert report.findings == []
    assert report.overall_reliability_score == 1.0


async def test_pipeline_all_agents_fail():
    with (
        patch(
            "orchestrator.agent_document_parser",
            new=AsyncMock(return_value=EMPTY_PARSED),
        ),
        patch(
            "orchestrator.agent_citation_verifier",
            new=AsyncMock(side_effect=RuntimeError("a")),
        ),
        patch(
            "orchestrator.agent_quote_verifier",
            new=AsyncMock(side_effect=RuntimeError("b")),
        ),
        patch(
            "orchestrator.agent_cross_document_checker",
            new=AsyncMock(side_effect=RuntimeError("c")),
        ),
        patch(
            "orchestrator.agent_judicial_memo_synthesizer",
            new=AsyncMock(return_value=EMPTY_MEMO),
        ),
    ):
        report = await run_pipeline(SAMPLE_DOCS)

    assert len(report.pipeline_errors) == 3
    assert report.findings == []
