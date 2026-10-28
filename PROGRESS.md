# PROGRESS.md — FiQA Financial RAG Portfolio Project

> **How to use:** This file is updated after every phase. Each item is marked
> `✅ DONE`, `🔄 IN PROGRESS`, `⏳ TODO`, or `❌ BLOCKED`. Never delete old
> entries — mark them done and add a date + commit hash.

---

## Phase 0 — Repo Foundation & Audit

| # | Task | Status | Commit / Notes |
|---|------|--------|----------------|
| 0.1 | Git init + `.gitignore` | ✅ DONE | Commit `93d561d` + ignore rules for SQLite, pickle, virtualenv |
| 0.2 | Clean `requirements.txt` (direct deps only) | ✅ DONE | Cleaned to 16 direct core dependencies |
| 0.3 | `PROGRESS.md` created | ✅ DONE | Living tracker created |
| 0.4 | `AUDIT.md` created | ✅ DONE | Baseline audit completed with evidence |
| 0.5 | `PROJECT_KNOWLEDGE.md` written | ✅ DONE | Comprehensive knowledge base with data topology & all milestones |
| 0.6 | Data integrity check (`src/data/verify_data.py`) | ✅ DONE | SHA-256 hashes, null checks, duplicates, collection verification |
| 0.7 | Reproduce key results with fixed seeds | ✅ DONE | `reproduce_baseline.py` executed; whitespace normalization bug solved |
| 0.8 | `portfolio-build` branch created | ✅ DONE | Active working branch |

---

## Phase 1 — Fill the Gaps (Retrieval & Generation Eval)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | BM25 standalone eval (nDCG@10, Recall@10/100, MRR) | ✅ DONE | `src/evaluation/retrieval_eval.py` (nDCG@10: 0.173 [0.079, 0.280]) |
| 1.2 | Dense retrieval eval (same metrics) | ✅ DONE | MiniLM (nDCG 0.377) & BGE (nDCG 0.440 [0.311, 0.570]) |
| 1.3 | Hybrid (RRF) eval | ✅ DONE | Fused RRF K=100 (Recall@100: 90.0% [80.0%, 100.0%]) |
| 1.4 | Hybrid + CrossEncoder reranker eval | ✅ DONE | Champion nDCG@10: 0.490 [0.348, 0.636]; MRR: 0.464 |
| 1.5 | Candidate pool sweep K=8/12/16 in notebook | ✅ DONE | Phase 9 added to notebook; K=8 proved Pareto optimal |
| 1.6 | Acronym query expansion test | ✅ DONE | `FinancialAcronymExpander` (+3.88 pts on HELOC, +1.44 pts on HSA/FSA) |
| 1.7 | 50-question golden set for generation eval | ✅ DONE | `data/golden_eval_set.json` (50 items) |
| 1.8 | RAGAS / Claim faithfulness + answer relevancy | ✅ DONE | `src/evaluation/generation_eval.py` (Faithfulness 96.8%, Relevancy 95.0%) |
| 1.9 | 95% bootstrap CIs on all metrics | ✅ DONE | `src/evaluation/metrics.py` (1,000 bootstrap resamples) |
| 1.10 | Ablation table (chunk size, top-k, reranker, prompt) | ✅ DONE | Candidate depth, Reranker on/off, Acronym ablation verified |

---

## Phase 2 — MLflow Pipeline

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | `src/pipeline/` stage functions | ⏳ TODO | |
| 2.2 | MLflow experiment tracking | ⏳ TODO | |
| 2.3 | `make pipeline` end-to-end | ⏳ TODO | |
| 2.4 | Import legacy results as MLflow runs | ⏳ TODO | |

---

## Phase 3 — FastAPI API

| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | `GET /health` | ✅ DONE | `app.py` |
| 3.2 | `POST /v1/query` with retriever selector | ✅ DONE | `app.py` |
| 3.3 | `GET /config` endpoint | ⏳ TODO | |
| 3.4 | Structured JSON logging | ⏳ TODO | |
| 3.5 | Request timeout handling | ⏳ TODO | |
| 3.6 | Pydantic v2 validation on all inputs | ✅ DONE | `schemas.py` |

---

## Phase 4 — Streamlit UI

| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | Question box + retriever selector | ⏳ TODO | |
| 4.2 | Answer with cited sources + latency | ⏳ TODO | |
| 4.3 | "Compare retrievers" view | ⏳ TODO | |
| 4.4 | "Evaluation results" tab | ⏳ TODO | |

---

## Phase 5 — Tests (pytest)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 5.1 | Unit: schemas validation | ✅ DONE | `tests/test_schemas.py` — 5/5 |
| 5.2 | Unit: LRU cache (set/get, TTL, LRU eviction) | ✅ DONE | `tests/test_cache.py` — 4/4 |
| 5.3 | Unit: guardrails (gate, citation) | ✅ DONE | `tests/test_guardrails.py` — 5/5 |
| 5.4 | Unit: chunking function | ⏳ TODO | |
| 5.5 | Unit: RRF fusion | ⏳ TODO | |
| 5.6 | Unit: pipeline stage caching | ⏳ TODO | |
| 5.7 | Data: no split overlap, no empty docs | ⏳ TODO | |
| 5.8 | API: health, valid query, 422, abstention (mock LLM) | ⏳ TODO | |
| 5.9 | Eval regression: nDCG@10 baseline guard | ⏳ TODO | |
| 5.10 | Coverage report ≥ 70% | ⏳ TODO | |

**Current: 14 / 14 existing tests passing ✅**

---

## Phase 6 — Packaging & CI/CD

| # | Task | Status | Notes |
|---|------|--------|-------|
| 6.1 | `Makefile` (setup, data, eval, test, run, docker-up) | ⏳ TODO | |
| 6.2 | `Dockerfile` (multi-stage, non-root) | ⏳ TODO | |
| 6.3 | `docker-compose.yml` (API + UI + MLflow) | ⏳ TODO | |
| 6.4 | GitHub Actions: lint + test + Docker build | ⏳ TODO | |
| 6.5 | CI uploads eval report as artifact | ⏳ TODO | |

---

## Phase 7 — Documentation

| # | Task | Status | Notes |
|---|------|--------|-------|
| 7.1 | `README.md` with real metrics + CIs | 🔄 IN PROGRESS | Needs metric update |
| 7.2 | `PROJECT_KNOWLEDGE.md` | ⏳ TODO | |
| 7.3 | `docs/PIPELINE.md` | ⏳ TODO | |
| 7.4 | `docs/RESULTS.md` | ⏳ TODO | |
| 7.5 | `docs/ARCHITECTURE.md` | ⏳ TODO | |
| 7.6 | `docs/TESTING.md` | ⏳ TODO | |
| 7.7 | `docs/INTERVIEW_NOTES.md` (25+ Q&A) | ⏳ TODO | |
| 7.8 | `AUDIT.md` kept up to date | 🔄 IN PROGRESS | |

---

## Stretch — French Multilingual Demo

| # | Task | Status | Notes |
|---|------|--------|-------|
| S.1 | 30 French FAQ/policy docs | ⏳ TODO | Only after all above done |
| S.2 | Multilingual embedding model | ⏳ TODO | |
| S.3 | 20 French questions + evaluation | ⏳ TODO | |

---

## Definition of Done ✅

From a **fresh clone**, all of these must work:
```bash
make setup       # installs deps
make data        # downloads + verifies data
make pipeline    # runs end-to-end
make test        # all tests pass
docker compose up  # API + UI + MLflow all healthy
```

Numbers in docs **must** match pipeline output. No TODOs. No placeholders.

---

*Last updated: 2026-10-28 | Phase: 0 — Git Foundation*

