# BS Detector — Design Decisions

This document explains the reasoning behind every significant choice made during implementation. The goal is to make each decision defensible in 30 seconds: what was chosen, why, and what the production alternative would be.

---

## 1. Agent decomposition — why five agents, structured this way

### DocumentParser as a single extraction agent (not three)

The first instinct is to split extraction into a CitationExtractor, a QuoteExtractor, and an AssertionExtractor. That decomposition adds two LLM calls and three coordination points for a task that fits comfortably in one prompt. The MSJ is a short document (~1,500 tokens). A single, well-structured JSON-mode call extracts all three categories reliably and in one round trip.

The rule applied: agents should have distinct *verification* roles, not distinct *parsing* roles, when the data fits in context. Splitting extraction is premature abstraction at this scale.

### Three parallel verification agents (not one)

Verification was split into CitationVerifier, QuoteVerifier, and CrossDocumentChecker because they operate on different corpora, require different domain knowledge, and fail in different ways:

- CitationVerifier needs legal knowledge about whether a case exists and whether the stated proposition matches the holding. Its inputs are compact (a list of citations). Its primary risk is hallucinating case knowledge.
- QuoteVerifier needs to detect word-level alterations in direct quotes. Its primary risk is missing subtle changes.
- CrossDocumentChecker needs verbatim evidence from the police report, medical records, and witness statement. Its primary risk is confabulating document text.

Merging these into one agent would produce an enormously long prompt with three distinct failure modes interleaved. Splitting them makes each agent's scope narrow, its failure mode isolated, and its prompt easier to reason about. It also makes them independently testable.

### JudicialMemoSynthesizer as the final stage

The synthesizer was kept as a separate agent rather than asking the last verification agent to "also write a summary." Summaries require a global view of all findings — selecting and ranking across citation, quote, and cross-document findings together. No single verification agent has that view. Separation of concerns here is not premature: it is the minimum necessary.

### The Orchestrator is not an agent

The orchestrator is a plain Python function. It sequences the agents, collects results, and computes the reliability score. There is no reason to make it an LLM agent — it does no reasoning, only coordination. Keeping it as code makes it fast, deterministic, and trivial to test.

---

## 2. Async execution — asyncio.gather for Step 2

The three verification agents in Step 2 have no dependencies on each other. They all depend on Step 1 (the parsed document) and are all consumed by Step 3 (the memo synthesizer). The obvious choice is to run them in parallel.

`asyncio.gather` with `return_exceptions=True` achieves this without threads and without aborting on partial failure. If CitationVerifier fails (e.g., due to an API timeout), `return_exceptions=True` returns the exception as a value instead of raising it. The `collect_results` function then extracts the exception into `pipeline_errors` and continues with whatever findings were produced.

`call_llm` is synchronous (the OpenAI SDK's sync client). `async_call_llm` wraps it with `asyncio.to_thread`, which runs it in a thread pool. This provides genuine I/O parallelism: all three API calls are in-flight simultaneously, not queued.

The practical result: Step 2 takes as long as the slowest of the three agents, not the sum of all three.

---

## 3. JSON mode for structured LLM output

Every agent uses `response_format={"type": "json_object"}` on the OpenAI call. This guarantees the LLM returns valid JSON, eliminates the most common failure mode (malformed output), and allows direct Pydantic validation with `model_validate`.

The alternative — asking the LLM to return JSON in regular text mode and parsing it — fails silently: the model may emit markdown fences, trailing commas, or truncated JSON. JSON mode eliminates all of that at zero cost.

Each agent's system prompt explicitly describes the JSON schema. This redundancy (schema in the prompt AND Pydantic validation on the output) is intentional: the prompt guides the model toward the right structure, and Pydantic catches any deviation that slips through. Failures at the Pydantic layer raise a `ValidationError`, which the orchestrator captures in `pipeline_errors`.

---

## 4. Hallucination prevention

Hallucination is the central risk in any LLM-based verification pipeline. Three mechanisms address it:

**Verbatim excerpts required.** Every `Evidence` object has a mandatory `excerpt` field. For any finding backed by a source document (not `legal_knowledge`), the agent must paste the exact text. The eval harness exploits this: it checks whether the excerpt actually appears in the source document using substring matching. If an agent "quotes" something the document never said, the eval catches it.

**Explicit `NOT_VERIFIABLE` verdict.** Every agent has `NOT_VERIFIABLE` as a valid finding type and `COULD_NOT_VERIFY` as a valid verdict. The prompts instruct the model to use these rather than invent a finding. Explicitly giving the model permission to say "I don't know" reduces confabulation more than simply telling it "don't hallucinate."

**Citation verifier uses explicit uncertainty language.** The citation verifier's prompt explicitly names real cases (Privette, Seabright) so the model does not need to fabricate knowledge about them. For unknown citations, the prompt instructs the model to flag them as `POSSIBLY_FABRICATED_CASE` rather than pretend to know they exist.

---

## 5. Data contracts — why Pydantic schemas across agent boundaries

Agents communicate via typed Pydantic models, not raw strings or dicts. This matters for three reasons:

**Correctness.** An agent that receives a `list[Citation]` cannot accidentally process a raw string. Type errors surface at the boundary, not deep inside the agent's logic.

**Testability.** Each agent can be unit-tested by constructing a valid Pydantic input and asserting on a Pydantic output. No orchestrator is needed for agent-level tests.

**Evolvability.** Adding a field to `Citation` breaks all agents that depend on `Citation` at validation time, not silently. In production, this forces intentional versioning of the data contract.

The `ParsedDocument → Finding → JudicialMemo → VerificationReport` chain makes the data flow explicit and the entire pipeline auditable from a single JSON blob.

---

## 6. Reliability score — deterministic, not LLM-generated

`overall_reliability_score` is computed by the orchestrator as a weighted penalty over finding severities:

```
score = 1.0 - (critical × 0.20 + high × 0.10 + medium × 0.05 + low × 0.01)
```

The alternative — asking the memo synthesizer to estimate a reliability score — would produce a number that varies between runs, is not auditable, and cannot be tested deterministically. A formula is less "intelligent" but more honest: it says exactly what it measures (severity-weighted penalty) and changes only when the findings change.

In production, this formula would be replaced by a calibrated model once ground-truth labels are available for historical cases.

---

## 7. Ground truth and eval design

### What the ground truth contains

Five known issues were identified by reading the four case documents:

| ID | Type | Issue |
|---|---|---|
| gt_1 | FACT_CONTRADICTION | MSJ states incident on March 14, 2021; all three source documents say March 12, 2021 |
| gt_2 | FACT_CONTRADICTION | MSJ claims Rivera was not wearing PPE; police report and witness statement both confirm he was |
| gt_3 | QUOTE_MISMATCH | Privette quote at page 702 ("A hirer is never liable…") is fabricated or materially altered; the actual holding establishes a rebuttable presumption, not an absolute rule |
| gt_4 | MISSING_EVIDENCE | MSJ argues Apex controlled the work; police report and witness statement show Harmon's foreman Donner directed the crew to the east-side section |
| gt_5 | MISSING_EVIDENCE | MSJ omits that Tran directly warned Donner about the defective base plate before the collapse and was dismissed |

### Metrics

**Precision** = correct findings / total findings reported. Measures how much of what the pipeline reports is actually true.

**Recall** = known issues found / total known issues. Measures whether the pipeline catches what matters.

**Hallucination rate** = unmatched findings / total findings reported. A finding is hallucinated if it does not correspond to any known issue by type and key phrase. This is the inverse of precision but named separately because it is the primary safety metric for a legal product.

**Matching logic:** A finding matches a known issue if (a) the `finding_type` is identical and (b) at least one key phrase from the ground-truth entry appears in the finding's `reasoning` or `subject_summary`. This is a conservative matching rule — it can miss valid findings that use different language. That is intentional: borderline matches should not inflate recall.

### Why not use an LLM as the eval judge

An LLM judge introduces a second LLM call per finding, doubles latency, and cannot be re-run deterministically. The string-matching approach is crude but reproducible: the same input always produces the same score. That reproducibility is worth more than marginal accuracy at eval time.

In production, the judge approach becomes necessary when ground truth is labeled with rich annotations and the pipeline output is prose rather than structured JSON. At this stage, structured output makes deterministic eval sufficient.

---

## 8. What was deliberately not built

| Decision | Reason |
|---|---|
| RAG / vector retrieval | All four documents total ~3,000 tokens. Passing them directly in context is simpler, cheaper, and more accurate than any retrieval scheme at this scale. |
| Streaming / SSE | The request-response model is sufficient for a 15–30 second demo. Streaming adds frontend complexity and a WebSocket or SSE layer without improving correctness. |
| Background jobs / polling | Same reasoning — acceptable for a prototype. Production alternative: a job queue with `/jobs/{id}/status`. |
| Auth | Out of scope for a single-tenant demo. In production: JWT + tenant-scoped document access before a second user touches the system. |
| Retry logic on LLM calls | The OpenAI SDK handles transient errors. Adding custom retry logic duplicates it. |
| Multi-model routing | All agents use `gpt-4o` at `temperature=0`. The consistency benefit (same model, same config) outweighs any cost optimization from routing cheaper tasks to smaller models — especially when correctness is the priority. |

---

## 9. What would change first in production

If this pipeline were moving toward production with real customers, the first two changes would be:

1. **Move the pipeline to a background worker.** The request-response model breaks at 30+ seconds. The fix is to enqueue a job on `POST /analyze`, return a job ID immediately, and let the client poll `GET /jobs/{id}`. The orchestrator code does not change — only where it runs.

2. **Add the Westlaw / CourtListener lookup for citation verification.** Right now, citation verification relies on model knowledge. A real legal product cannot ship that. The CitationVerifier agent is designed to swap in an external API call — its interface (`list[Citation] → list[Finding]`) stays the same; only the evidence source changes from `legal_knowledge` to a live lookup.

Everything else (auth, multi-tenancy, observability, streaming progress) follows in roughly that priority order.
