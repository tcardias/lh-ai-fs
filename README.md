# BS Detector

A multi-agent AI pipeline that verifies legal briefs against supporting documents. Given a Motion for Summary Judgment, it extracts citations, checks factual consistency, flags inaccurate quotes, and produces a structured verification report with a judge-facing memo.

The demo case is *Rivera v. Harmon Construction Group* (Case No. BC-2023-04851).

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env      # Set OPENAI_API_KEY=<your-key>
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8002 |
| UI  | http://localhost:5175 |

Both services hot-reload on file changes.

### Manual

**Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # Set OPENAI_API_KEY
uvicorn main:app --reload --port 8002
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

---

## Running the Eval Suite

The eval script must run inside the backend virtualenv (dependencies are installed there, not system-wide):

```bash
cd backend
source venv/bin/activate   # Windows: venv\Scripts\activate
python run_evals.py
```

Or without activating the venv:

```bash
cd backend
venv/bin/python run_evals.py
```

Output is printed to stdout and saved to `backend/eval_results.json`.

The harness measures **precision**, **recall**, and **hallucination rate** against five known ground-truth issues in the demo MSJ (`backend/ground_truth.json`). See [`docs/decisions.md`](docs/decisions.md) for how the metrics are defined.

---

## Running the Test Suite

The test suite requires no API key — all LLM calls are mocked.

```bash
cd backend
source venv/bin/activate          # Windows: venv\Scripts\activate
pytest tests/ -v
```

`pytest` e `pytest-asyncio` já estão em `requirements.txt`.

| File | What it covers |
|------|----------------|
| `tests/test_models.py` | Pydantic validation rules, boundary values, invalid enum values |
| `tests/test_orchestrator.py` | Reliability scoring, partial agent failures, full pipeline with mocked agents |
| `tests/test_evals.py` | Precision/recall/hallucination metrics, phrase matching, ground-truth structure |

---

## API

### `POST /analyze`

No request body required — documents are loaded server-side from `backend/documents/`.

**Response**

```json
{
  "report": {
    "case_id": "BC-2023-04851",
    "analyzed_document": "motion_for_summary_judgment",
    "supporting_documents": ["police_report", "medical_records_excerpt", "witness_statement"],
    "parsed": { "citations": [...], "quotes": [...], "factual_assertions": [...] },
    "findings": [...],
    "judicial_memo": "...",
    "top_findings": [...],
    "overall_reliability_score": 0.65,
    "pipeline_errors": []
  }
}
```

---

## Project Structure

```
.
├── backend/
│   ├── main.py                        # FastAPI app — POST /analyze entry point
│   ├── orchestrator.py                # Pipeline coordinator (asyncio.gather)
│   ├── models.py                      # All Pydantic schemas
│   ├── llm.py                         # OpenAI wrapper (call_llm, async_call_llm)
│   ├── agents/
│   │   ├── document_parser.py         # Step 1 — extract citations, quotes, assertions
│   │   ├── citation_verifier.py       # Step 2a — verify legal citations
│   │   ├── quote_verifier.py          # Step 2b — verify direct quotes
│   │   ├── cross_document_checker.py  # Step 2c — compare MSJ facts vs. source docs
│   │   └── judicial_memo_synthesizer.py # Step 3 — judge-facing summary
│   ├── documents/
│   │   ├── motion_for_summary_judgment.txt
│   │   ├── police_report.txt
│   │   ├── medical_records_excerpt.txt
│   │   └── witness_statement.txt
│   ├── ground_truth.json              # Known issues for eval
│   ├── run_evals.py                   # Eval harness (precision / recall / hallucination)
│   ├── requirements.txt               # All dependencies (runtime + test)
│   ├── pyproject.toml                 # Ruff and pytest configuration
│   ├── conftest.py                    # Pytest sys.path setup
│   └── tests/
│       ├── test_models.py             # Pydantic validation tests
│       ├── test_orchestrator.py       # Pipeline flow and scoring tests
│       └── test_evals.py             # Eval metric tests
├── frontend/
│   └── src/
│       └── App.jsx                    # Single-page report viewer
├── docs/
│   ├── decisions.md                   # Design decisions and implementation rationale
│   ├── architecture-plan.md           # Agent design and data contracts
│   ├── production-readiness-plan.md        # Production MVP plan
│   └── README-original.md            # Original challenge README
└── docker-compose.yml
```

---

## Pipeline Overview

```
POST /analyze
      │
      ▼
 Orchestrator
      │
      ├─ Step 1 ──► DocumentParser
      │                 (extract citations, quotes, factual assertions from MSJ)
      │
      ├─ Step 2 ──► [parallel — asyncio.gather]
      │              ├─ CitationVerifier
      │              ├─ QuoteVerifier
      │              └─ CrossDocumentChecker
      │
      └─ Step 3 ──► JudicialMemoSynthesizer
                        │
                        ▼
                  VerificationReport (JSON)
```

Agent failures in Step 2 are captured in `pipeline_errors` and do not abort the pipeline.

---

## Further Reading

- [`docs/decisions.md`](docs/decisions.md) — design decisions, agent prompting rationale, eval metric design, explicit tradeoffs
- [`docs/architecture-plan.md`](docs/architecture-plan.md) — full data contracts and agent specifications
- [`docs/production-readiness-plan.md`](docs/production-readiness-plan.md) — production MVP plan
- [`docs/README-original.md`](docs/README-original.md) — original challenge brief
