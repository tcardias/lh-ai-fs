"""Tests for the eval harness: metric calculations and matching logic."""

import pytest

from models import (
    Confidence,
    Evidence,
    Finding,
    FindingType,
    ParsedDocument,
    Severity,
    Verdict,
    VerificationReport,
)
from run_evals import _finding_matches_issue, evaluate, load_ground_truth

# ── Helpers ────────────────────────────────────────────────────────────────────


def make_finding(finding_id: str, finding_type: str, reasoning: str = "") -> Finding:
    return Finding(
        finding_id=finding_id,
        finding_type=FindingType(finding_type),
        severity=Severity.HIGH,
        verdict=Verdict.CONTRADICTED,
        subject_id="fa_1",
        subject_summary=reasoning,
        reasoning=reasoning,
        evidence=[Evidence(source_document="legal_knowledge", excerpt="", relevance="x")],
        confidence=Confidence(score=0.9, explanation="high"),
    )


def make_report(findings: list[Finding]) -> VerificationReport:
    return VerificationReport(
        supporting_documents=[],
        parsed=ParsedDocument(citations=[], quotes=[], factual_assertions=[]),
        findings=findings,
        judicial_memo="",
        top_findings=[],
        overall_reliability_score=1.0,
        pipeline_errors=[],
    )


GT_ONE_ISSUE = {
    "known_issues": [
        {
            "id": "gt_1",
            "finding_type": "FACT_CONTRADICTION",
            "description": "Wrong date",
            "key_phrases": ["March 12", "March 14"],
        }
    ]
}


# ── _finding_matches_issue ─────────────────────────────────────────────────────


def test_match_correct_type_and_phrase():
    f = make_finding("f_1", "FACT_CONTRADICTION", "incident on March 12 not March 14")
    assert _finding_matches_issue(f, GT_ONE_ISSUE["known_issues"][0]) is True


def test_no_match_wrong_type():
    f = make_finding("f_1", "QUOTE_MISMATCH", "incident on March 12 not March 14")
    assert _finding_matches_issue(f, GT_ONE_ISSUE["known_issues"][0]) is False


def test_no_match_no_phrase():
    f = make_finding("f_1", "FACT_CONTRADICTION", "completely unrelated reasoning")
    assert _finding_matches_issue(f, GT_ONE_ISSUE["known_issues"][0]) is False


def test_match_is_case_insensitive():
    f = make_finding("f_1", "FACT_CONTRADICTION", "MARCH 12 was the actual date")
    assert _finding_matches_issue(f, GT_ONE_ISSUE["known_issues"][0]) is True


def test_match_phrase_in_subject_summary():
    f = Finding(
        finding_id="f_1",
        finding_type=FindingType.FACT_CONTRADICTION,
        severity=Severity.HIGH,
        verdict=Verdict.CONTRADICTED,
        subject_id="fa_1",
        subject_summary="Date is wrong: March 14 vs March 12",
        reasoning="unrelated",
        evidence=[Evidence(source_document="legal_knowledge", excerpt="", relevance="x")],
        confidence=Confidence(score=0.9, explanation="high"),
    )
    assert _finding_matches_issue(f, GT_ONE_ISSUE["known_issues"][0]) is True


# ── evaluate ──────────────────────────────────────────────────────────────────


def test_evaluate_perfect_recall_and_precision():
    f = make_finding("f_1", "FACT_CONTRADICTION", "incident on March 12 not March 14")
    results = evaluate(make_report([f]), GT_ONE_ISSUE)
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["hallucination_rate"] == 0.0
    assert results["f1_score"] == pytest.approx(1.0)


def test_evaluate_no_findings():
    results = evaluate(make_report([]), GT_ONE_ISSUE)
    assert results["precision"] == 0.0
    assert results["recall"] == 0.0
    assert results["total_findings_reported"] == 0


def test_evaluate_only_hallucinations():
    f = make_finding("f_1", "FACT_CONTRADICTION", "completely unrelated")
    results = evaluate(make_report([f]), GT_ONE_ISSUE)
    assert results["hallucination_rate"] == 1.0
    assert results["recall"] == 0.0
    assert results["precision"] == 0.0


def test_evaluate_mixed_correct_and_hallucinated():
    correct = make_finding("f_1", "FACT_CONTRADICTION", "March 12 March 14")
    hallucinated = make_finding("f_2", "FACT_CONTRADICTION", "unrelated noise")
    results = evaluate(make_report([correct, hallucinated]), GT_ONE_ISSUE)
    assert results["correct_findings"] == 1
    assert results["hallucinated_findings"] == 1
    assert results["precision"] == pytest.approx(0.5)
    assert results["recall"] == 1.0


def test_evaluate_same_issue_not_double_counted():
    """Two findings matching the same ground-truth issue should count as one recall hit."""
    f1 = make_finding("f_1", "FACT_CONTRADICTION", "March 12 March 14")
    f2 = make_finding("f_2", "FACT_CONTRADICTION", "March 12 March 14 again")
    results = evaluate(make_report([f1, f2]), GT_ONE_ISSUE)
    assert results["known_issues_found"] == 1
    assert results["recall"] == 1.0


# ── load_ground_truth ──────────────────────────────────────────────────────────


def test_load_ground_truth_structure():
    gt = load_ground_truth()
    assert "known_issues" in gt
    assert len(gt["known_issues"]) == 5
    for issue in gt["known_issues"]:
        assert "id" in issue
        assert "finding_type" in issue
        assert "key_phrases" in issue
        assert len(issue["key_phrases"]) > 0
