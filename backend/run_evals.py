"""
Evaluation harness for the BS Detector pipeline.

Metrics:
  precision  = correct_findings / total_findings_reported
  recall     = known_issues_found / total_known_issues
  hallucination_rate = hallucinated_findings / total_findings_reported

A finding is "correct" if it matches a known issue by finding_type AND
at least one key_phrase from the ground truth appears in the finding's
reasoning or subject_summary (case-insensitive).

A finding is "hallucinated" if it does not match any known issue at all.

Run:
  python run_evals.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from models import Finding, VerificationReport  # noqa: E402
from orchestrator import run_pipeline  # noqa: E402


def load_documents() -> dict[str, str]:
    """Load all .txt case documents from the documents/ directory."""
    docs_dir = Path(__file__).parent / "documents"
    return {p.stem: p.read_text() for p in docs_dir.glob("*.txt")}


def load_ground_truth() -> dict[str, Any]:
    """Load the ground-truth issue definitions from ground_truth.json."""
    gt_path = Path(__file__).parent / "ground_truth.json"
    return json.loads(gt_path.read_text())


def _finding_matches_issue(finding: Finding, issue: dict[str, Any]) -> bool:
    """Return True if a finding corresponds to a ground-truth issue.

    Matching requires both:
    - The ``finding_type`` value is identical to the issue's ``finding_type``.
    - At least one ``key_phrase`` from the issue appears (case-insensitive) in
      the finding's ``reasoning`` or ``subject_summary``.
    """
    if finding.finding_type.value != issue["finding_type"]:
        return False
    haystack = (finding.reasoning + " " + finding.subject_summary).lower()
    return any(phrase.lower() in haystack for phrase in issue["key_phrases"])


def evaluate(report: VerificationReport, ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Compute precision, recall, hallucination rate, and F1 for a pipeline report.

    Args:
        report: The ``VerificationReport`` produced by ``run_pipeline``.
        ground_truth: Parsed content of ``ground_truth.json``.

    Returns:
        A dict with keys: ``total_findings_reported``, ``correct_findings``,
        ``hallucinated_findings``, ``known_issues_total``, ``known_issues_found``,
        ``precision``, ``recall``, ``hallucination_rate``, ``f1_score``,
        ``overall_reliability_score``, ``pipeline_errors``, ``missed_issues``.
    """
    known_issues: list[dict[str, Any]] = ground_truth["known_issues"]
    findings = report.findings

    matched_issue_ids: set[str] = set()
    correct_finding_ids: set[str] = set()

    for finding in findings:
        for issue in known_issues:
            if issue["id"] not in matched_issue_ids and _finding_matches_issue(finding, issue):
                matched_issue_ids.add(issue["id"])
                correct_finding_ids.add(finding.finding_id)
                break

    total_findings = len(findings)
    correct = len(correct_finding_ids)
    hallucinated = total_findings - correct
    known_found = len(matched_issue_ids)
    total_known = len(known_issues)

    precision = correct / total_findings if total_findings else 0.0
    recall = known_found / total_known if total_known else 0.0
    hallucination_rate = hallucinated / total_findings if total_findings else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    missed_issues = [i for i in known_issues if i["id"] not in matched_issue_ids]

    return {
        "total_findings_reported": total_findings,
        "correct_findings": correct,
        "hallucinated_findings": hallucinated,
        "known_issues_total": total_known,
        "known_issues_found": known_found,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "f1_score": round(f1, 4),
        "overall_reliability_score": report.overall_reliability_score,
        "pipeline_errors": report.pipeline_errors,
        "missed_issues": [i["id"] + ": " + i["description"][:80] for i in missed_issues],
    }


def print_results(results: dict[str, Any]) -> None:
    """Print a formatted eval summary to stdout."""
    print("\n" + "=" * 60)
    print("BS DETECTOR — EVAL RESULTS")
    print("=" * 60)
    print(f"  Findings reported:     {results['total_findings_reported']}")
    print(f"  Correct findings:      {results['correct_findings']}")
    print(f"  Hallucinated findings: {results['hallucinated_findings']}")
    print(f"  Known issues total:    {results['known_issues_total']}")
    print(f"  Known issues found:    {results['known_issues_found']}")
    print()
    print(f"  Precision:             {results['precision']:.2%}")
    print(f"  Recall:                {results['recall']:.2%}")
    print(f"  Hallucination rate:    {results['hallucination_rate']:.2%}")
    print(f"  F1 score:              {results['f1_score']:.4f}")
    print(f"  Reliability score:     {results['overall_reliability_score']:.3f}")
    if results["pipeline_errors"]:
        print(f"\n  Pipeline errors ({len(results['pipeline_errors'])}):")
        for err in results["pipeline_errors"]:
            print(f"    - {err}")
    if results["missed_issues"]:
        print(f"\n  Missed known issues ({len(results['missed_issues'])}):")
        for issue in results["missed_issues"]:
            print(f"    - {issue}")
    print("=" * 60 + "\n")


async def main() -> None:
    """Load documents, run the pipeline, evaluate, and persist results."""
    print("Loading documents...")
    documents = load_documents()
    ground_truth = load_ground_truth()

    print("Running pipeline...")
    report = await run_pipeline(documents)

    print(f"Pipeline produced {len(report.findings)} findings.")
    results = evaluate(report, ground_truth)
    print_results(results)

    output_path = Path(__file__).parent / "eval_results.json"
    output_path.write_text(json.dumps({"eval": results, "report": report.model_dump()}, indent=2, default=str))
    print(f"Full results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
