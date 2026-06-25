"""Tests for Pydantic model validation rules."""

import pytest
from pydantic import ValidationError

from models import (
    FactualAssertion,
    Finding,
    FindingType,
    ParsedDocument,
    Severity,
    Verdict,
    VerificationReport,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _confidence(**overrides) -> dict:
    return {"score": 0.9, "explanation": "High certainty", **overrides}


def _evidence(**overrides) -> dict:
    return {
        "source_document": "legal_knowledge",
        "excerpt": "",
        "relevance": "Relevant to the finding",
        **overrides,
    }


def _finding(**overrides) -> dict:
    return {
        "finding_id": "f_test_1",
        "finding_type": "FACT_CONTRADICTION",
        "severity": "high",
        "verdict": "CONTRADICTED",
        "subject_id": "fa_1",
        "subject_summary": "Test assertion",
        "reasoning": "The documents contradict this.",
        "evidence": [_evidence()],
        "confidence": _confidence(),
        **overrides,
    }


def _empty_parsed() -> ParsedDocument:
    return ParsedDocument(citations=[], quotes=[], factual_assertions=[])


# ── Finding validation ─────────────────────────────────────────────────────────


def test_finding_valid():
    f = Finding.model_validate(_finding())
    assert f.finding_type == FindingType.FACT_CONTRADICTION
    assert f.severity == Severity.HIGH
    assert f.verdict == Verdict.CONTRADICTED


def test_finding_invalid_finding_type_raises():
    """'CONFIRMED' is a Verdict value, not a FindingType — must be rejected."""
    with pytest.raises(ValidationError):
        Finding.model_validate(_finding(finding_type="CONFIRMED"))


def test_finding_unknown_finding_type_raises():
    with pytest.raises(ValidationError):
        Finding.model_validate(_finding(finding_type="NONSENSE"))


def test_finding_all_valid_types():
    valid_types = [
        "FACT_CONTRADICTION",
        "QUOTE_MISMATCH",
        "UNSUPPORTED_CITATION",
        "POSSIBLY_FABRICATED_CASE",
        "LEGAL_OVERSTATEMENT",
        "MISSING_EVIDENCE",
        "NOT_VERIFIABLE",
    ]
    for ft in valid_types:
        f = Finding.model_validate(_finding(finding_type=ft))
        assert f.finding_type.value == ft


# ── Confidence validation ──────────────────────────────────────────────────────


def test_confidence_score_above_one_raises():
    with pytest.raises(ValidationError):
        Finding.model_validate(_finding(confidence=_confidence(score=1.1)))


def test_confidence_score_negative_raises():
    with pytest.raises(ValidationError):
        Finding.model_validate(_finding(confidence=_confidence(score=-0.1)))


def test_confidence_score_boundary_values():
    Finding.model_validate(_finding(confidence=_confidence(score=0.0)))
    Finding.model_validate(_finding(confidence=_confidence(score=1.0)))


# ── Evidence validation ────────────────────────────────────────────────────────


def test_evidence_valid_source_documents():
    valid_sources = [
        "police_report",
        "medical_records_excerpt",
        "witness_statement",
        "motion_for_summary_judgment",
        "legal_knowledge",
    ]
    for source in valid_sources:
        f = Finding.model_validate(_finding(evidence=[_evidence(source_document=source)]))
        assert f.evidence[0].source_document == source


def test_evidence_invalid_source_raises():
    with pytest.raises(ValidationError):
        Finding.model_validate(_finding(evidence=[_evidence(source_document="unknown_doc")]))


# ── ParsedDocument ─────────────────────────────────────────────────────────────


def test_parsed_document_accepts_empty_lists():
    doc = _empty_parsed()
    assert doc.citations == []
    assert doc.quotes == []
    assert doc.factual_assertions == []


def test_factual_assertion_valid_types():
    for assertion_type in (
        "date",
        "equipment",
        "employment",
        "legal_status",
        "procedural",
        "other",
    ):
        FactualAssertion(
            assertion_id="fa_1",
            msj_section="II",
            text="Some fact.",
            assertion_type=assertion_type,
        )


def test_factual_assertion_invalid_type_raises():
    with pytest.raises(ValidationError):
        FactualAssertion(
            assertion_id="fa_1",
            msj_section="II",
            text="Some fact.",
            assertion_type="invalid",
        )


# ── VerificationReport ─────────────────────────────────────────────────────────


def test_verification_report_defaults():
    report = VerificationReport(
        supporting_documents=["police_report"],
        parsed=_empty_parsed(),
        findings=[],
        judicial_memo="No issues found.",
        top_findings=[],
        overall_reliability_score=1.0,
        pipeline_errors=[],
    )
    assert report.case_id == "BC-2023-04851"
    assert report.analyzed_document == "motion_for_summary_judgment"


def test_verification_report_reliability_above_one_raises():
    with pytest.raises(ValidationError):
        VerificationReport(
            supporting_documents=[],
            parsed=_empty_parsed(),
            findings=[],
            judicial_memo="",
            top_findings=[],
            overall_reliability_score=1.5,
            pipeline_errors=[],
        )


def test_verification_report_reliability_negative_raises():
    with pytest.raises(ValidationError):
        VerificationReport(
            supporting_documents=[],
            parsed=_empty_parsed(),
            findings=[],
            judicial_memo="",
            top_findings=[],
            overall_reliability_score=-0.1,
            pipeline_errors=[],
        )
