# AUDIT.md — FiQA Financial RAG Portfolio Build

> This document provides an honest, empirical record of what existed before the portfolio-build phase, what was changed and why, all assumptions made, and the architectural justification for all major technical decisions.
> **Absolute Honesty Rule:** Never invent or estimate numbers — every metric cited here is grounded in real commands executed on the BEIR FiQA benchmark.

---

## 1. Existing Artifacts (Baseline Audit)

### Data
| File | Status | Empirical Verification |
|---|---|---|
| `data/raw/fiqa_corpus.csv` | ✅ VERIFIED | 57,638 raw rows $\rightarrow$ 38 null text rows dropped $\rightarrow$ **57,598 unique valid documents**. |
| `data/raw/fiqa_eval.csv` | ✅ VERIFIED | 30-query academic benchmark with ground truth contexts (`contexts`). |
| `data/raw/fiqa_eval_new.json` | ✅ VERIFIED | 45-query enterprise stress test with domain categories and target doc IDs. |
| `data/golden_eval_set.json` | ✅ VERIFIED | 50-item hand-checkable golden evaluation set (45 from eval_new + 5 edge cases). |
| `data/chroma_db/` | ✅ VERIFIED | ChromaDB persistent store with 59,926 chunks across `fiqa_full_corpus` (MiniLM) and `fiqa_bge_corpus` (BGE-small). |
| `data/bm25_index.pkl` | ✅ VERIFIED | In-memory tokenized BM25Okapi index (91.34 MB) over all 59,926 chunks. |

### Source Code Modules
| Module | Status | Notes |
|---|---|---|
| `src/config.py` | ✅ VERIFIED | Immutable `Settings` dataclass with all empirically verified hyperparameters. |
| `src/schemas.py` | ✅ VERIFIED | Pydantic V2 schemas: `QueryRequest`, `QueryResponse`, `SourceItem`, `GateStatus`, `HealthResponse`, `ConfigResponse`. |
| `src/cache.py` | ✅ VERIFIED | Thread-safe LRU cache (`OrderedDict`) with 1-hour TTL, 1000 max size. |
| `src/data/chunker.py` | ✅ VERIFIED | Sliding window text chunking (500 chars, 50 overlap) with boundary protection. |
| `src/retrieval/dense.py` | ✅ VERIFIED | ChromaDB vector query engine for MiniLM and BGE embeddings. |
| `src/retrieval/sparse.py` | ✅ VERIFIED | BM25Okapi tokenized lexical query engine. |
| `src/retrieval/hybrid.py` | ✅ VERIFIED | Fused hybrid candidate pool + Reciprocal Rank Fusion ($1/(60+r)$). |
| `src/retrieval/query_expansion.py` | ✅ VERIFIED | `FinancialAcronymExpander` mapping 19 core regulatory abbreviations. |
| `src/ranking/reranker.py` | ✅ VERIFIED | Cross-encoder joint-attention neural re-ranker (`ms-marco-MiniLM-L-6-v2`). |
| `src/guardrails/confidence_gate.py` | ✅ VERIFIED | Confidence Gate circuit breaker ($\tau=0.0$). |
| `src/guardrails/citation_auditor.py` | ✅ VERIFIED | Sentence-level citation extraction and precision calculation. |
| `src/generation/llm.py` | ✅ VERIFIED | Gemini 3.5 Flash Lite defensive prompt generator with exponential backoff. |
| `src/pipeline/orchestrator.py` | ✅ VERIFIED | Unified pipeline orchestrator supporting all 4 retriever modes. |
| `src/pipeline/stages.py` | ✅ VERIFIED | 7 idempotent stages with SHA-256 `.cache_manifest.json` caching. |
| `src/pipeline/tracker.py` | ✅ VERIFIED | MLflow experiment tracking and legacy experiment ingestion. |
| `src/pipeline/cli.py` | ✅ VERIFIED | CLI dispatcher (`all`, `eval`, `stage <name>`, `import-legacy`). |
| `app.py` | ✅ VERIFIED | FastAPI REST API with structured JSON logging, config endpoint, and 30s timeout SLA. |
| `ui/app.py` | ✅ VERIFIED | Streamlit interactive UI with Live Query, Retriever Comparison, and Benchmark Dashboard. |

---

## 2. Test Suite & Code Quality Audit

| Test File | Tests | Coverage | Status |
|---|---|---|---|
| `tests/test_schemas.py` | 5 | 100% | ✅ PASSING |
| `tests/test_cache.py` | 4 | 94% | ✅ PASSING |
| `tests/test_guardrails.py` | 5 | 100% | ✅ PASSING |
| `tests/test_metrics.py` | 6 | 88% | ✅ PASSING |
| `tests/test_query_expansion.py` | 3 | 100% | ✅ PASSING |
| `tests/test_retrieval_and_chunking.py` | 5 | 100% | ✅ PASSING |
| `tests/test_data_integrity.py` | 4 | 100% | ✅ PASSING |
| `tests/test_eval_regression.py` | 1 | 100% | ✅ PASSING (Baseline: 0.6708, Tol: 0.05) |
| `tests/test_api.py` | 9 | 88% | ✅ PASSING |
| `tests/test_pipeline_stages.py` | 5 | 100% | ✅ PASSING |
| `tests/test_coverage_boost.py` | 8 | 100% | ✅ PASSING |
| **Total** | **55** | **73% Total** | **✅ 100% PASS RATE** |

---

## 3. Empirically Verified Metrics & Baseline Reproduction

### Dual Embedding Collection Audit
| Metric | `all-MiniLM-L6-v2` (`fiqa_full_corpus`) | `BAAI/bge-small-en-v1.5` (`fiqa_bge_corpus`) | Relative Lift with BGE |
|---|---|---|---|
| **Hit@1 (Top-1 Accuracy)** | 26.7% (8/30) | **46.7% (14/30)** | **+75.0% relative lift** |
| **Hit@3 (Top-3 Recall)** | 46.7% (14/30) | **60.0% (18/30)** | **+28.5% relative lift** |
| **Hit@5 (Candidate Recall)** | 60.0% (18/30) | **66.7% (20/30)** | **+11.2% relative lift** |
| **MRR** | 0.395 | **0.531** | **+34.4% relative lift** |

### Academic Retrieval Benchmark on BEIR FiQA Test Set
| Architecture | nDCG@10 (95% CI) | Recall@10 (95% CI) | Recall@100 (95% CI) | MRR (95% CI) | P95 Latency |
|---|---|---|---|---|---|
| **BM25 Only** | 0.173 [0.079, 0.280] | 30.0% [13.3%, 46.7%] | 56.7% [36.7%, 76.7%] | 0.183 [0.088, 0.292] | 423 ms |
| **Dense (MiniLM)** | 0.377 [0.257, 0.490] | 70.0% [53.3%, 86.7%] | 100.0% [100.0%, 100.0%] | 0.409 [0.281, 0.536] | 1,021 ms |
| **Dense (BGE-Small)** | 0.440 [0.311, 0.570] | 70.0% [53.3%, 86.7%] | 96.7% [90.0%, 100.0%] | **0.565 [0.428, 0.704]** | 1,180 ms |
| **Hybrid RRF** | 0.347 [0.231, 0.470] | 56.7% [36.7%, 73.3%] | 90.0% [80.0%, 100.0%] | 0.356 [0.236, 0.485] | 1,245 ms |
| **Two-Stage Champion** | **0.490 [0.348, 0.636]** | **70.0% [53.3%, 86.7%]** | 90.0% [80.0%, 100.0%] | 0.464 [0.323, 0.612] | 1,374 ms |

### Root-Cause Finding on Whitespace Matching Discrepancy
When evaluating raw substring containment without whitespace normalization (`\s+`), raw double spaces in the forum scraped FiQA dataset (e.g. `"deposit it.  It's called"`) caused exact chunk matches (`"deposit it. It's called"`) to register as false negatives, artificially depressing Hit@1 from 43.3% to 13.3%. Normalizing whitespace via `re.sub(r'\s+', ' ', text)` resolved this and restored the true accuracy.

---

## 4. Architectural Decisions & Assumptions Log

| ID | Decision / Assumption | Rationale | Empirical Verification |
|---|---|---|---|
| **A1** | **ChromaDB Dual Collection** | MiniLM stored in `fiqa_full_corpus`; BGE-small stored in `fiqa_bge_corpus` | Verified: BGE provides +75.0% relative lift in Hit@1 (46.7% vs 26.7%) |
| **A2** | **Airflow DAG vs Native CLI Pipeline** | **Airflow skipped:** Running Airflow locally requires Postgres, Redis, Celery/Local workers, Webserver, and Scheduler containers (>4GB RAM overhead). | Our 7-stage idempotent CLI (`src/pipeline/cli.py`) with SHA-256 config-hash caching delivers sub-second skipping without heavyweight infrastructure. |
| **A3** | **Candidate Pool Depth ($K=8$)** | Swept $K \in [8, 12, 16]$ | $K=8$ proved Pareto-optimal; $K=16$ increased latency by +74.5% with zero Hit@1 gain and slight MRR dilution |
| **A4** | **Acronym Query Expansion** | Financial queries are saturated with abbreviations (`HELOC`, `HSA`, `FSA`) | Deterministic regex expansion lifted mean confidence by +100.5% and rescued false-negative circuit breaker trips |
| **A5** | **Confidence Gate ($\tau=0.0$)** | Halts out-of-domain queries before LLM dispatch | Achieved 94.0% gate accuracy, blocking 100% of out-of-domain traps at $0.00 token cost |

---

## 5. Complete Record of Portfolio Changes

| Date | Phase | Changes Landed | Commit |
|---|---|---|---|
| 2026-10-28 | **Phase 0** | Git init, `.gitignore`, clean direct-deps `requirements.txt`, baseline reproduction script, `PROGRESS.md`, `AUDIT.md`, `PROJECT_KNOWLEDGE.md` | `93d561d`, `9a5e9fe` |
| 2026-10-28 | **Phase 1** | Full BEIR retrieval evaluation, 1,000 bootstrap resamples for 95% CIs, 50-item golden set (`golden_eval_set.json`), acronym expander, candidate depth sweep | `5d17ea5` |
| 2026-10-28 | **Phase 2** | 7-stage idempotent pipeline in `src/pipeline/`, MLflow experiment tracker (`tracker.py`), legacy run importer, cross-platform `Makefile`, `docs/PIPELINE.md` | `b2e3f0a` |
| 2026-10-28 | **Phase 3** | FastAPI multi-retriever routing (`/query`), runtime config endpoint (`/config`), structured JSON logging middleware, 30s request timeout SLA (HTTP 504) | `ce209b5` |
| 2026-10-28 | **Phase 4** | Interactive Streamlit interface (`ui/app.py`) with Live Query, side-by-side retriever comparison, and academic evaluation dashboard | `4d91620` |
| 2026-10-28 | **Phase 5** | Test suite expanded to 55 tests, hand-computed RRF verification, sliding window chunking tests, data hygiene tests, nDCG@10 regression guard, 73% code coverage | `69a1c8b` |
| 2026-10-28 | **Phase 6** | Hardened multi-stage Dockerfile (non-root `appuser`, UID 10001), `docker-compose.yml`, GitHub Actions CI workflow (`.github/workflows/ci.yml`), `docker-down` target | `c72692c` |
| 2026-10-28 | **Phase 7** | Comprehensive documentation suite: `docs/RESULTS.md`, `docs/ARCHITECTURE.md`, `docs/TESTING.md`, `docs/INTERVIEW_NOTES.md` (30 Q&A), updated `README.md` | `2036c56` |
| 2026-10-28 | **Stretch** | French multilingual demo (`src/multilingual/`), 30 shop FAQ docs, `paraphrase-multilingual-MiniLM-L12-v2` dense vectors + BM25 RRF, 20 queries eval, 7 tests | Active |
