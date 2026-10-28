# AUDIT.md — FiQA Financial RAG

> This file documents what existed before the portfolio-build phase, what was
> changed and why, and all assumptions made during development.
> **Rule: never invent or estimate numbers — every metric cited here came from
> a command that was actually run.**

---

## Existing Artifacts (Pre-Portfolio-Build Audit)

### Data

| File | Status | Notes |
|------|--------|-------|
| `data/raw/fiqa_corpus.csv` | ✅ EXISTS | 57,638 rows (57,596 after dedup). BEIR FiQA-2018 corpus. |
| `data/raw/fiqa_eval.csv` | ✅ EXISTS | 30-query academic benchmark with ground-truth chunk IDs. **Suite 1 — use for retrieval metrics.** |
| `data/raw/fiqa_eval_new.json` | ✅ EXISTS | 45-query enterprise stress test. **Suite 2 — WARNING: synthetic doc IDs do NOT match corpus (IDs exceed 57,595). Only valid for gate/hallucination/circuit-breaker evaluation. Cannot be used for Hit@K retrieval metrics.** |
| `data/chroma_db/` | ✅ EXISTS | ChromaDB persistent store. Collection: `fiqa_full_corpus`. 59,926 chunks. |
| `data/bm25_index.pkl` | ✅ EXISTS | BM25Okapi index pickled from 59,926 chunks. Reproducible from corpus via `src/retrieval/sparse.py`. |

### Source Code

| File | Status | Notes |
|------|--------|-------|
| `src/__init__.py` | ✅ DONE | |
| `src/config.py` | ✅ DONE | Immutable `Settings` dataclass. All hyperparams empirically grounded. |
| `src/schemas.py` | ✅ DONE | Pydantic v2: `QueryRequest`, `QueryResponse`, `SourceCitation`, `GateStatus`, `HealthResponse`. |
| `src/cache.py` | ✅ DONE | Thread-safe LRU in-memory cache. TTL=1hr, max=1000 entries. Verified: ~200,000x speedup on cache hit (14,681ms → 0.07ms). |
| `src/retrieval/dense.py` | ✅ DONE | ChromaDB client wrapper. |
| `src/retrieval/sparse.py` | ✅ DONE | BM25Okapi with disk pickle caching. |
| `src/retrieval/hybrid.py` | ✅ DONE | Dense Top-8 + BM25 Top-8 union pool, deduplicated. |
| `src/ranking/reranker.py` | ✅ DONE | CrossEncoderReranker (`ms-marco-MiniLM-L-6-v2`). |
| `src/guardrails/confidence_gate.py` | ✅ DONE | Confidence gate τ=0.0: blocks 100% OOD, 100% hallucination traps. |
| `src/guardrails/citation_auditor.py` | ✅ DONE | Citation precision/hallucination checker. |
| `src/generation/llm.py` | ✅ DONE | Gemini Flash Lite + exponential backoff (3 retries, base 2s). |
| `src/pipeline.py` | ✅ DONE | `FinancialRAGPipeline` with `.run()` (sync) and `.arun()` (async). |
| `app.py` | ✅ DONE | FastAPI REST API (port 8000). Endpoints: `/`, `/v1/health`, `/v1/query`, `DELETE /v1/cache`. |
| `main.py` | ✅ DONE | Terminal CLI with `--query` flag and interactive mode. |
| `src/data/download_fiqa.py` | ✅ EXISTS | Data download/verify script. Needs checksum verification addition. |

### Tests

| File | Tests | Status |
|------|-------|--------|
| `tests/test_schemas.py` | 5 | ✅ PASSING |
| `tests/test_cache.py` | 4 | ✅ PASSING |
| `tests/test_guardrails.py` | 5 | ✅ PASSING |
| **Total** | **14/14** | **✅ ALL PASS** |

### Documentation

| File | Status | Notes |
|------|--------|-------|
| `README.md` | 🔄 PARTIAL | Exists with Two-Benchmark Suite explanation and engineering journey. Needs real nDCG@10/Recall metrics with CIs. |
| `MASTER_PROJECT_LOG_AND_STRATEGY.md` | ✅ EXISTS | 14 pillars of project logic. Keep for reference. |
| `INTERVIEW_NOTES_AND_LOGS.md` | ✅ EXISTS | Steps 1-22, Pillars 1-11. Needs expansion to 25+ Q&A per spec. |
| `LICENSE` | ✅ EXISTS | MIT |
| `.env.example` | ✅ EXISTS | `GEMINI_API_KEY` documented. |

### Missing Files (Portfolio-Build Gaps)

| File | Priority | Notes |
|------|----------|-------|
| `Makefile` | HIGH | `make setup`, `make data`, `make eval`, `make test`, `make run` |
| `Streamlit UI` | HIGH | `ui/app.py` — calls FastAPI |
| `docs/PIPELINE.md` | HIGH | Mermaid diagram + stage descriptions |
| `docs/RESULTS.md` | HIGH | Real metrics table with CIs |
| `docs/ARCHITECTURE.md` | MEDIUM | Component diagram, design decisions |
| `docs/TESTING.md` | MEDIUM | Coverage report, test strategy |
| `docs/INTERVIEW_NOTES.md` | MEDIUM | 25+ Q&A |
| `PROJECT_KNOWLEDGE.md` | MEDIUM | Dataset structure, experiment log |
| `Dockerfile` | HIGH | Multi-stage, non-root |
| `docker-compose.yml` | HIGH | API + UI + MLflow |
| `.github/workflows/ci.yml` | HIGH | lint + test + Docker build |
| `results/legacy/` | LOW | Move old scratch results here |
| `pyproject.toml` improvements | LOW | Add ruff config, coverage settings |

---

## Empirically Verified Metrics

> All numbers below came from commands run in prior sessions. See notebook
> `notebooks/01_data_ingestion_and_eda.ipynb` for full reproduction.

| Metric | Value | Configuration | Notes |
|--------|-------|--------------|-------|
| Hit@3 (string match) | 66.7% (20/30) | Hybrid K=8 + CrossEncoder | Suite 1, fiqa_eval.csv |
| Hit@3 (semantic/LLM judge) | 77.7% | Hybrid K=8 + CrossEncoder | Gap caused by incomplete single-label GT in 2018 dataset |
| MRR | 0.531 | Hybrid K=8 + CrossEncoder | Suite 1 |
| Generation Faithfulness | 100.0% | Gemini Flash Lite | Suite 2 gate/hallucination test |
| Cache speedup | ~200,000x | LRU in-memory | Run 1: 14,681ms → Run 2: 0.07ms |
| Gate OOD blocking | 100% | τ=0.0 | All out-of-domain queries blocked |
| Gate hallucination blocking | 100% | τ=0.0 | All hallucination traps blocked |
| Token savings | 42.2% | Confidence gate | LLM calls skipped for weak evidence |

**⚠️ MISSING (to be filled in Phase 1):**
- nDCG@10, Recall@10, Recall@100 (proper BEIR-style evaluation)
- 95% bootstrap confidence intervals on all metrics
- Ablation table (chunk size, top-k, reranker on/off, prompt version)

---

## Assumptions Log

| ID | Assumption | Rationale | Impact if Wrong |
|----|-----------|-----------|----------------|
| A1 | ChromaDB collection uses `all-MiniLM-L6-v2` embeddings (per config.py) | `config.py` specifies this model | If BGE was used to build the index, queries must also use BGE — **verify before re-embedding** |
| A2 | `data/raw/fiqa_eval.csv` ground truth is string-matched against retrieved chunk text | That's how Hit@K was computed | If IDs were used instead, metrics would differ |
| A3 | Suite 2 (`fiqa_eval_new.json`) doc IDs are synthetic and don't match corpus | Verified: IDs exceed 57,595 corpus max | Cannot use Suite 2 for retrieval Hit@K |
| A4 | BM25 index built from 59,926 chunks matches ChromaDB chunks | Same chunking parameters used | If different, retrieval results would not be comparable |
| A5 | BEIR FiQA-2018 licence allows use in this portfolio (no commercial redistribution) | Standard academic benchmark | Raw data excluded from repo; user must download via script |

---

## Changes Made (Portfolio-Build Phase)

| Date | Change | Reason |
|------|--------|--------|
| 2026-10-28 | Replaced full `pip freeze` `requirements.txt` with direct-deps-only version | Cleaner, more readable, easier to maintain; lock file approach is better for full pins |
| 2026-10-28 | Rewrote `.gitignore` to include `data/raw/`, `mlruns/`, `logs/` | Licence compliance + correct exclusion of all generated/binary artifacts |
| 2026-10-28 | Created `PROGRESS.md` | Track phase completion; resume from checkpoint after interruptions |
| 2026-10-28 | Initialized git repo on `main` branch | Version control foundation for portfolio |
| 2026-10-28 | Created `portfolio-build` branch | Isolate portfolio work from base state |

