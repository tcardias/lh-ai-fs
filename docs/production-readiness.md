# BS Detector — Production Readiness Plan

## Scope and Intent

This document describes how to move BS Detector from the current working prototype to a paid MVP that can serve law firms and legal teams at startup scale. It is specific to the domain, the pipeline, and the product risks of a legal AI verification tool — not a generic SaaS scaling template.

The prototype today is a single `POST /analyze` endpoint that loads documents from disk, runs a multi-agent LLM pipeline synchronously, and returns a report. That is fine for a demo. It does not survive real users.

---

## Assumptions

| Dimension | Assumption | Reasoning |
|-----------|-----------|-----------|
| Launch scale | Hundreds of concurrent users | Consistent with the brief; manageable with a single job queue |
| Growth target | Tens of thousands of users | Horizontal scaling path must be clear, but not over-engineered at MVP |
| Documents per matter | 10–200 documents, 5–200 pages each | A large patent dispute or personal injury case can exceed this; we treat it as P95 |
| Analysis latency | 2–10 minutes per analysis | Acceptable for legal work; users are not waiting at a keyboard |
| LLM cost per analysis | $0.50–$3.00 (gpt-4o at current pricing) | Must be factored into pricing; caching and batching are levers |
| Reliability SLA | 99.5% job completion | One failure in 200 is tolerable; silent wrong answers are not |
| Data sensitivity | Attorney-client privilege, work product | This is the highest-risk axis; drives every security decision |
| Regulatory scope | US courts for MVP | Avoid EU data residency complexity at launch; defer to growth |

If document volume grows beyond 200 per matter, the cross-document checker needs a retrieval layer (vector search) instead of full-context injection. That threshold is the main architectural fork in the road.

---

## What Changes Between Prototype and MVP

The prototype has three hard blockers for production:

1. **Synchronous pipeline on the HTTP thread** — a 5-minute LLM chain will time out in any load balancer or client.
2. **Documents read from local disk** — breaks with any horizontal scaling or user-uploaded files.
3. **No tenant isolation** — every caller sees the same data.

Everything else in the prototype — the agent decomposition, the Pydantic contracts, the parallel `asyncio.gather` — is worth keeping.

---

## System Components

```
┌─────────────┐     upload      ┌──────────────────┐
│   Browser   │ ─────────────► │   API Gateway     │
│  (React UI) │                 │  (FastAPI + auth) │
└─────────────┘                 └────────┬─────────┘
                                         │
                          enqueue job    │
                                         ▼
                                ┌──────────────────┐      ┌─────────────────┐
                                │   Job Queue      │      │  Object Store   │
                                │   (Redis/SQS)    │      │  (S3 / R2)      │
                                └────────┬─────────┘      └────────▲────────┘
                                         │ dequeue                  │ read doc
                                         ▼                          │
                                ┌──────────────────┐               │
                                │  Worker Pool     │───────────────┘
                                │  (Celery/ARQ)    │
                                │  ├ DocumentParser│
                                │  ├ CitationVerif │      ┌─────────────────┐
                                │  ├ QuoteVerifier │─────►│   LLM Provider  │
                                │  ├ CrossDocCheck │      │   (OpenAI API)  │
                                │  └ MemoSynth     │      └─────────────────┘
                                └────────┬─────────┘
                                         │ store result
                                         ▼
                                ┌──────────────────┐
                                │   PostgreSQL      │
                                │  (jobs, reports,  │
                                │   tenants, audit) │
                                └──────────────────┘
```

**API Gateway** — FastAPI handles auth (JWT + API keys), rate limiting, and job submission. No business logic here; it translates HTTP into queue messages.

**Object Store** — Raw documents live in S3 (or Cloudflare R2 for cost). One prefix per tenant: `s3://{bucket}/{tenant_id}/{matter_id}/{doc_id}`. Workers read directly from S3; the API never proxies document bytes.

**Job Queue** — Redis-backed (Celery or ARQ). Each analysis is a single job; the worker fetches documents from S3, runs the pipeline, and writes the report to Postgres. The queue handles retries, dead-letter, and concurrency limits.

**Worker Pool** — Stateless containers running the existing pipeline code. Horizontal scaling is straightforward: add workers. Workers do not share state.

**PostgreSQL** — Source of truth for job state, reports, tenant config, and the audit log. Reports are stored as JSONB; no separate document database needed at MVP scale.

---

## Data Flow: Upload to Report

```
1. User uploads MSJ + supporting docs via browser
2. API authenticates, creates a Matter record, uploads files to S3
3. API enqueues AnalysisJob(matter_id, tenant_id, doc_ids)
4. API returns { job_id, status: "queued" }
5. UI polls GET /jobs/{job_id} every 10s (or uses SSE/WebSocket)
6. Worker picks up job, fetches docs from S3
7. Worker runs pipeline (DocumentParser → parallel agents → MemoSynth)
8. Worker writes VerificationReport to Postgres, sets status=completed
9. UI fetches report, renders findings
```

Status transitions: `queued → running → completed | failed`

Partial failures (one agent errors) set `pipeline_errors` and still complete the job with partial results. A full pipeline crash sets status=`failed` with a retry-eligible flag.

---

## Data Durability

| Data | Storage | Rationale |
|------|---------|-----------|
| Raw documents | S3 (versioned) | Source of truth; never recomputed; attorney files must be preserved |
| VerificationReport JSON | Postgres JSONB | Queried, diffed, audited; must be durable |
| LLM intermediate outputs | Discarded after report | Can be recomputed; storing them inflates cost and attack surface |
| Job status | Postgres | Needs to survive worker crash |
| Audit log | Postgres append-only table | Never updated or deleted; required for legal accountability |
| Vector embeddings (future) | pgvector or Pinecone | Recomputable from raw docs; can be rebuilt on model upgrade |

The audit log records every document access, every analysis run, every report view, with tenant_id, user_id, timestamp, and job_id. This is non-negotiable for legal clients.

---

## Tenant Isolation

Each tenant gets:

- A **tenant_id** UUID stamped on every database row
- A **separate S3 prefix** (`/{tenant_id}/`); bucket policies deny cross-prefix reads
- An **API key** scoped to that tenant; the API injects tenant_id from the key, never from the request body
- **Row-level security** in Postgres: `ALTER TABLE matters ENABLE ROW LEVEL SECURITY; CREATE POLICY tenant_isolation ON matters USING (tenant_id = current_setting('app.tenant_id')::uuid);`

Workers receive tenant_id in the job payload and set the Postgres session variable before any query. No worker query can see another tenant's rows.

LLM prompts never embed one tenant's documents in another tenant's context. Each job fetches only the documents listed in its payload.

---

## Where the System Fails First

**#1: LLM rate limits**

At hundreds of concurrent analyses, each firing 5 serial or parallel gpt-4o calls, we will hit OpenAI's TPM/RPM limits quickly. Mitigations in priority order:

1. Per-tenant concurrency limit in the job queue (e.g. max 3 active jobs per tenant)
2. Exponential backoff with jitter on `429` responses (already partially in place)
3. OpenAI Batch API for lower-priority analyses (50% cost reduction, 24h SLA)
4. Second LLM provider (Anthropic Claude) on failover for non-latency-sensitive steps

**#2: Cross-document checker with large matters**

The current implementation injects all source documents into a single prompt (~3k tokens for the demo case). A 50-document matter blows the context window. Fix: chunk documents, embed them, and retrieve top-K relevant chunks per assertion before sending to the agent. This requires a vector store (pgvector is sufficient at MVP scale) and an ingestion step at upload time.

This is the most significant architectural change from prototype to MVP.

**#3: Runaway LLM costs**

A single malicious or misconfigured request could trigger dozens of expensive LLM calls. Controls:

- Per-tenant monthly spend cap (tracked via job cost estimates stored in Postgres)
- Per-job document size limit (e.g. 500KB total before vector retrieval)
- Hard timeout on workers (e.g. 15 minutes max); jobs that exceed it are failed and flagged

**#4: Worker crash mid-pipeline**

The pipeline is not yet idempotent. If a worker dies after CitationVerifier but before QuoteVerifier, the job is re-queued and re-runs from scratch (wasting cost). MVP mitigation: checkpoint intermediate agent results to Postgres so a retry can skip completed steps. Full implementation deferred to post-MVP.

---

## Security

**Transport**: TLS everywhere. API keys in `Authorization: Bearer` header, never in query strings or logs.

**Storage**: S3 SSE-S3 encryption at rest (upgrade to SSE-KMS for customers requiring key management). Postgres TDE via managed service (RDS, Supabase).

**LLM provider**: Opt out of OpenAI training via the API (default for API customers). Do not log prompt content server-side beyond what the audit log requires. Consider Azure OpenAI for customers requiring data residency.

**Prompt injection**: User-supplied document content is injected into LLM prompts. We must treat all document text as untrusted. The existing prompts already wrap document content in explicit delimiters (`--- BEGIN DOCUMENT ---`). Add a preprocessing step that strips known injection patterns (e.g. "Ignore previous instructions") from document text before it enters any prompt.

**Least privilege**: Workers run with an IAM role that can only read from `s3://{bucket}/{tenant_id}/` (scoped to their job's tenant). The API role can write to S3 but cannot read arbitrary prefixes.

---

## Observability

Three layers matter for a legal AI product:

**System health**
- Job queue depth and age (alert if p99 wait > 5 minutes)
- Worker error rate by agent
- LLM API error rate and latency by model
- Cost per job (tracked in Postgres, surfaced in a simple admin dashboard)

**Pipeline correctness**
- The eval harness (`run_evals.py`) runs against every deployment against the ground-truth dataset. Precision and recall regressions block deploys.
- Human review queue: a random sample (5%) of completed reports is flagged for attorney review. Confirmed false positives/negatives feed back into the ground-truth dataset.
- `pipeline_errors` rate per tenant. A spike indicates a prompt regression or schema drift.

**Business metrics**
- Time-to-report by matter size
- Report re-run rate (a proxy for dissatisfied users)
- Findings per document type (tracks whether certain document categories are under-served)

No ML observability platform is needed at MVP. A Postgres table + Grafana (or Metabase) is sufficient.

---

## What to Build First

The prototype ships. The first production increment closes the three hard blockers, in order:

### Increment 1 — Unblock real users (2–3 weeks)

1. **Async job queue**: Replace synchronous `POST /analyze` response with a job submission pattern. Add `GET /jobs/{job_id}` polling endpoint. Use Redis + Celery (or ARQ if already on asyncio). The pipeline code does not change; it moves into a Celery task.
2. **S3 document storage**: Replace `load_documents()` disk reads with S3 fetches keyed by job payload. Add a `POST /matters/{id}/documents` upload endpoint.
3. **Tenant auth**: Add JWT-based auth with tenant_id claim. Apply row-level security in Postgres. Scope all queries.

This is all plumbing. No pipeline changes. At the end of Increment 1, the product can accept real customers.

### Increment 2 — Handle real matters (3–4 weeks)

4. **Vector retrieval for cross-document checker**: Embed documents at upload time (pgvector). Replace full-context injection with top-K retrieval per assertion. This is required for any matter beyond the demo case.
5. **LLM rate limit handling**: Per-tenant concurrency limits in queue + retry with backoff.
6. **Audit log**: Append-only Postgres table recording every job and document access.

### Deferred to Post-MVP

- **Citation database lookup**: Verifying case citations against a live legal database (Westlaw, CourtListener) requires a legal data partnership. High value, high cost, not needed to launch.
- **Fine-tuning / prompt caching**: Significant cost reduction, but requires a prompt stability baseline first.
- **Multi-model routing**: Use a cheaper model (gpt-4o-mini) for DocumentParser and a stronger model for CitationVerifier. Worth ~60% cost reduction, but adds complexity.
- **Streaming report delivery**: WebSockets for real-time finding updates as each agent completes. Good UX; low priority vs. correctness.
- **European data residency**: Required for EU clients; defer until a concrete customer demands it.

---

## What Should Stay Flexible

The product is early and the hardest questions about BS Detector are product questions, not engineering ones:

- **Which finding types matter most to attorneys?** The current taxonomy (7 types) is a hypothesis. The agents and the Pydantic models should be easy to add to or reclassify without schema migrations.
- **What confidence threshold triggers a "real" flag?** The reliability score formula is deterministic today. It will need tuning once real attorneys tell us which severities they care about.
- **How much human-in-the-loop is right?** The architecture should support adding a review step before a report is delivered to a customer, without rebuilding the pipeline.

The job queue and Pydantic schemas are the right places to absorb these changes. Keep those interfaces stable; keep everything else easy to modify.

---

## Risks Worth Calling Out

**Hallucination in a legal context is a liability event.** The existing pipeline requires verbatim excerpts as evidence in every finding, which reduces (not eliminates) fabricated citations. Before any paying customer uses production output, there must be a disclosure that this is an AI-assisted tool requiring attorney review, and the UI must display that caveat prominently.

**The ground-truth eval set is tiny.** Five issues against one demo document is enough to catch regressions during development; it is not enough to claim the system is accurate. A real accuracy claim requires a professionally annotated dataset of at least 50 matters. This is the most important thing to build that is not engineering.

**OpenAI dependency is a single point of failure and a compliance risk.** An outage stops all analyses. A policy change on data retention could create legal exposure for customers. The LLM abstraction layer (`llm.py:call_llm`) is already designed to swap providers; use it.
