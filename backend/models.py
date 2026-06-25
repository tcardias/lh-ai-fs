from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class FindingType(StrEnum):
    """Categories of problems the pipeline can detect in an MSJ."""

    FACT_CONTRADICTION = "FACT_CONTRADICTION"
    QUOTE_MISMATCH = "QUOTE_MISMATCH"
    UNSUPPORTED_CITATION = "UNSUPPORTED_CITATION"
    POSSIBLY_FABRICATED_CASE = "POSSIBLY_FABRICATED_CASE"
    LEGAL_OVERSTATEMENT = "LEGAL_OVERSTATEMENT"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


class Severity(StrEnum):
    """How serious a finding is for the motion's credibility."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Verdict(StrEnum):
    """The pipeline's conclusion about a specific item under review."""

    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    FABRICATED = "FABRICATED"
    ALTERED = "ALTERED"
    COULD_NOT_VERIFY = "COULD_NOT_VERIFY"
    CONFIRMED = "CONFIRMED"


class Citation(BaseModel):
    """A legal citation extracted from the MSJ."""

    citation_id: str
    case_name: str
    reporter: str
    year: int
    volume: int | None = None
    start_page: int | None = None
    pin_page: int | None = None
    court: str | None = None
    proposition: str
    msj_section: str


class Quote(BaseModel):
    """A direct quote attributed to a case or document in the MSJ."""

    quote_id: str
    attributed_to: str
    attributed_page: str | None = None
    quoted_text: str
    msj_context: str
    msj_section: str


class FactualAssertion(BaseModel):
    """A factual claim in the MSJ to be checked against source documents."""

    assertion_id: str
    msj_section: str
    text: str
    assertion_type: Literal["date", "equipment", "employment", "legal_status", "procedural", "other"]


class ParsedDocument(BaseModel):
    """Structured output of DocumentParser: all extractable elements of the MSJ."""

    citations: list[Citation]
    quotes: list[Quote]
    factual_assertions: list[FactualAssertion]


class Evidence(BaseModel):
    """A piece of supporting evidence for a finding, with a verbatim source excerpt."""

    source_document: Literal[
        "police_report",
        "medical_records_excerpt",
        "witness_statement",
        "motion_for_summary_judgment",
        "legal_knowledge",
    ]
    excerpt: str
    relevance: str


class Confidence(BaseModel):
    """LLM-provided certainty estimate for a finding."""

    score: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class Finding(BaseModel):
    """A single verified problem detected in the MSJ."""

    finding_id: str
    finding_type: FindingType
    severity: Severity
    verdict: Verdict
    subject_id: str
    subject_summary: str
    reasoning: str
    evidence: list[Evidence]
    confidence: Confidence


class TopFinding(BaseModel):
    """A condensed reference to a finding for the judicial memo."""

    finding_id: str
    one_line_summary: str
    severity: Severity
    confidence_score: float


class JudicialMemo(BaseModel):
    """Output of JudicialMemoSynthesizer: judge-facing summary and ranked findings."""

    text: str
    top_findings: list[TopFinding]


class VerificationReport(BaseModel):
    """The complete output of the BS Detector pipeline for one MSJ."""

    case_id: str = "BC-2023-04851"
    analyzed_document: str = "motion_for_summary_judgment"
    supporting_documents: list[str]
    parsed: ParsedDocument
    findings: list[Finding]
    judicial_memo: str
    top_findings: list[TopFinding]
    overall_reliability_score: float = Field(..., ge=0.0, le=1.0)
    pipeline_errors: list[str]
