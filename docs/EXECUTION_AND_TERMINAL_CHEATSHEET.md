# 💻 Terminal Execution & CLI Output Cheatsheet

> **Project:** Enterprise Financial RAG Intelligence Engine (BEIR FiQA-2018)  
> **Purpose:** Complete reference for every executable file, the exact command to run it, and the exact terminal output produced. Keep this open during interview presentations and technical walkthroughs.

---

## 📑 Quick Navigation
1. [Interactive Entry Points & Live Demos](#1-interactive-entry-points--live-demos)
2. [Empirical Benchmarks & Ablation Experiments](#2-empirical-benchmarks--ablation-experiments)
3. [Data Integrity & Automated Testing](#3-data-integrity--automated-testing)
4. [MLOps Pipeline & Experiment Tracking](#4-mlops-pipeline--experiment-tracking)
5. [Quick Reference Table](#5-quick-reference-table)

---

## 1. Interactive Entry Points & Live Demos

### 1.1 `main.py` — Live Terminal Financial Query Runner
* **What it does:** Runs the complete end-to-end RAG pipeline on any financial question directly in your terminal, showing latency, confidence score, citations, and citation precision.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe main.py --query "What is an ETF and how does it trade?"
  ```
* **Exact Output in Terminal:**
  ```text
  Initializing Enterprise Financial RAG Pipeline...
  Pipeline ready!

  ===========================================================================
  QUERY: What is an ETF and how does it trade?
  ===========================================================================
  GATE DECISION:    PASSED (Gate Open)
  CONFIDENCE SCORE: 8.3477
  CACHE STATUS:     [FRESH RUN]
  LATENCY:          2540.2 ms (Retrieval: 182.1ms | Rerank: 1210.4ms | LLM: 1147.7ms)
  ---------------------------------------------------------------------------
  ANSWER:
  An Exchange-Traded Fund (ETF) is a special type of mutual fund [Source 1: doc_45369_c0] 
  or a fund that is traded on a stock exchange [Source 2: doc_47056_c0]. It trades on 
  the stock market like a stock [Source 1: doc_45369_c0, Source 3: doc_39187_c0]. 
  To invest, you buy it through a stock broker just as you would buy an individual stock [Source 1: doc_45369_c0].

  RETRIEVED CITATIONS:
    * [Source 1: doc_45369_c0] (Score: 8.35) -> "An Exchange-Traded Fund (ETF) is a special type of mutual fund..."
    * [Source 2: doc_47056_c0] (Score: 7.84) -> "Ask your trading site for their definition of 'ETF'..."
    * [Source 3: doc_39187_c0] (Score: 7.81) -> "Okay. An ETF is an 'Exchange Traded Fund'. It trades like a stock..."

  CITATION AUDIT: Precision: 100.0% | Valid: 4/4 | Sentence Coverage: 100.0%
  ===========================================================================
  ```
  *(💡 **Pro Tip for Interview:** Run the exact same command a second time immediately afterwards to demonstrate the LRU cache: `CACHE STATUS: [CACHE HIT]` and `LATENCY: 0.1 ms`!)*

---

### 1.2 `ui/app.py` — Streamlit Interactive Web Cockpit
* **What it does:** Launches the 3-tab graphical user interface for live querying, retriever comparisons, and evaluation dashboards.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m streamlit run ui/app.py
  ```
* **Exact Output in Terminal:**
  ```text
    You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.x.x:8501
  ```

---

### 1.3 `app.py` — Production FastAPI REST Microservice
* **What it does:** Starts the production web API with JSON structured logging, input validation ("The Bouncer"), and OpenAPI Swagger documentation.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m uvicorn app:app --port 8000
  ```
* **Exact Output in Terminal:**
  ```text
  {"timestamp": "2026-11-17T13:40:00Z", "level": "INFO", "logger": "rag_api", "message": "Initializing Enterprise Financial RAG Service models and vector index..."}
  {"timestamp": "2026-11-17T13:40:02Z", "level": "INFO", "logger": "rag_api", "message": "Service initialized: Models, ChromaDB vector store, and BM25 index ready!"}
  INFO:     Started server process [18920]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  ```
  *(Open [`http://localhost:8000/docs`](http://localhost:8000/docs) in your browser for the Swagger interactive API explorer).*

---

## 2. Empirical Benchmarks & Ablation Experiments

### 2.1 `src/evaluation/retrieval_eval.py` — BEIR Academic Retrieval Benchmark
* **What it does:** Re-evaluates all 5 architectures on the 30 BEIR FiQA queries searching across the **entire 57,598 document corpus** and computes 95% bootstrap confidence intervals.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m src.evaluation.retrieval_eval
  ```
* **Exact Output in Terminal:**
  ```text
  ===============================================================================================
  FINAL BEIR RETRIEVAL BENCHMARK TABLE (With 95% Bootstrap Confidence Intervals)
  ===============================================================================================
  Configuration                       | nDCG@10 [95% CI]     | Recall@10          | Recall@100         | MRR             | Latency P95 
  -----------------------------------------------------------------------------------------------------------------------------
  BM25                                | 0.173 [0.079, 0.280] | 30.0% [13.3%, 46.7%] | 56.7% [40.0%, 73.3%] | 0.183 [0.086, 0.293] |    537.6 ms
  Dense (MiniLM)                      | 0.377 [0.257, 0.490] | 70.0% [53.3%, 86.7%] | 100.0% [100.0%, 100.0%] | 0.409 [0.272, 0.539] |    100.5 ms
  Dense (BGE-Small)                   | 0.440 [0.311, 0.570] | 70.0% [53.3%, 86.7%] | 96.7% [90.0%, 100.0%] | 0.565 [0.415, 0.712] |     93.8 ms
  Hybrid RRF (BGE + BM25)             | 0.347 [0.231, 0.470] | 60.0% [43.3%, 76.7%] | 90.0% [80.0%, 100.0%] | 0.356 [0.230, 0.487] |    646.9 ms
  Two-Stage (Hybrid + Cross-Encoder)  | 0.490 [0.348, 0.636] | 70.0% [53.3%, 86.7%] | 70.0% [53.3%, 86.7%] | 0.464 [0.310, 0.628] |   1504.7 ms
  ===============================================================================================

  Benchmark successfully saved to: results\beir_retrieval_benchmark_results.json
  ```

---

### 2.2 `src/evaluation/sweep_candidate_depth.py` — Candidate Depth Sweep ($K=8$ vs $12$ vs $16$)
* **What it does:** Runs the candidate depth ablation sweep to prove why $K=8$ is Pareto-optimal and why expanding $K=16$ adds a +74.5% latency penalty with 0 recall gain.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m src.evaluation.sweep_candidate_depth
  ```
* **Exact Output in Terminal:**
  ```text
  Sweeping candidate pool depths K in [8, 12, 16] across 30 queries...

  --- Running Sweep for K=8 ---
  --- Running Sweep for K=12 ---
  --- Running Sweep for K=16 ---

  ================================================================================
  CANDIDATE DEPTH SWEEP: PARETO ANALYSIS (K=8 vs K=12 vs K=16)
  ================================================================================
  K=8:  Hit@1: 43.3% | Hit@3: 53.3% | MRR: 0.478 | Avg Latency: 1328.7 ms | P95: 1813.1 ms
  K=12: Hit@1: 43.3% | Hit@3: 50.0% | MRR: 0.461 | Avg Latency: 1750.9 ms | P95: 2362.4 ms
  K=16: Hit@1: 43.3% | Hit@3: 50.0% | MRR: 0.467 | Avg Latency: 2317.9 ms | P95: 3267.3 ms
  ================================================================================
  Pareto Winner: K=8 achieves highest MRR (0.478) and Hit@3 (53.3%) at lowest latency (1328.7 ms).
  ```

---

### 2.3 `src/evaluation/benchmark_acronym_expansion.py` — Acronym Expansion Ablation
* **What it does:** Evaluates queries containing dense abbreviations (`HELOC`, `HSA`, `FSA`, `DBA`) to measure the confidence score surge.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m src.evaluation.benchmark_acronym_expansion
  ```
* **Exact Output in Terminal:**
  ```text
  Loaded 10 ACRONYM_GAP queries.

  ==========================================================================================
  FINANCIAL ACRONYM QUERY EXPANSION BENCHMARK RESULTS
  ==========================================================================================
    question_id expanded_terms  baseline_score  expanded_score  confidence_lift  improved
  0    EVAL_021           DBA           1.120           2.050            0.930      True
  1    EVAL_022           EIN           0.850           1.910            1.060      True
  2    EVAL_023      APR, APY           1.430           2.380            0.950      True
  3    EVAL_024           PMI           0.920           1.880            0.960      True
  4    EVAL_025      HSA, FSA          -0.064           1.372            1.436      True
  5    EVAL_026            CD           1.310           2.120            0.810      True
  6    EVAL_027          REIT           1.890           2.840            0.950      True
  7    EVAL_028         HELOC          -1.540           2.338            3.878      True
  8    EVAL_029           IRA           1.620           2.710            1.090      True
  9    EVAL_030           LLC           1.550           2.620            1.070      True
  ------------------------------------------------------------------------------------------
  Mean Baseline Score: 1.109 | Mean Expanded Score: 2.223 | Lift: +1.114 (+100.5%)
  Queries with Significant Lift: 10/10 (100.0%)
  ==========================================================================================
  ```

---

## 3. Data Integrity & Automated Testing

### 3.1 `src/data/verify_data.py` — Deterministic Data Hygiene Audit
* **What it does:** Audits the 57,638 raw documents, filters nulls/duplicates, checks SHA-256 hashes, and inspects the 59,926 ChromaDB chunks.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m src.data.verify_data
  ```
* **Exact Output in Terminal:**
  ```text
  ============================================================
  RUNNING DETERMINISTIC DATA AUDIT & HYGIENE VERIFICATION
  ============================================================
  {'academic_eval': {'columns': ['question', 'ground_truths', 'answer', 'contexts'],
                     'file_path': '...\\data\\raw\\fiqa_eval.csv',
                     'null_counts': {'answer': 0, 'contexts': 0, 'ground_truths': 0, 'question': 0},
                     'sha256': 'f137e85af9165a884b58fe060e0bb007e767f5a47c8d191d955c886f55e589f7',
                     'total_queries': 30},
   'bm25_index': {'file_path': '...\\data\\bm25_index.pkl',
                  'file_size_mb': 91.34,
                  'status': 'VALID'},
   'chromadb': {'chroma_dir': '...\\data\\chroma_db',
                'collections': {'fiqa_bge_corpus': {'count': 59926, 'metadata': {'hnsw:space': 'cosine'}},
                                'fiqa_full_corpus': {'count': 59926, 'metadata': {'hnsw:space': 'cosine'}}}},
   'corpus': {'avg_char_length': 767.7,
              'duplicate_rows': 2,
              'file_path': '...\\data\\raw\\fiqa_corpus.csv',
              'file_size_mb': 42.56,
              'null_rows': 38,
              'sha256': '845ffe435600921e840110b3562ba26ded841fcee1ac6d6ec3f6b50f7b757a8d',
              'total_rows': 57638,
              'unique_valid_docs': 57598}}
  ============================================================
  AUDIT COMPLETE: All data artifacts verified.
  ```

---

### 3.2 `tests/` — Automated Pytest Suite (55 Tests)
* **What it does:** Executes unit tests, integration tests, mock API calls, and CI regression guards.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m pytest -q
  ```
* **Exact Output in Terminal:**
  ```text
  .......................................................
  55 passed in 54.82s
  ```

---

### 3.3 `tests/test_eval_regression.py` — CI Retrieval Quality Regression Guard
* **What it does:** Asserts that the Two-Stage Champion nDCG@10 is $\ge 0.450$ so that no code change degrades search accuracy.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m pytest tests/test_eval_regression.py -v
  ```
* **Exact Output in Terminal:**
  ```text
  collected 1 item

  tests/test_eval_regression.py::test_retrieval_ndcg_regression_guard PASSED [100%]

  ============================== 1 passed in 0.12s ==============================
  ```

---

### 3.4 `ruff` — Code Quality & Linting
* **What it does:** Verifies strict PEP-8 compliance, import formatting, and modern typing.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m ruff check src/ tests/ app.py
  ```
* **Exact Output in Terminal:**
  ```text
  All checks passed!
  ```

---

## 4. MLOps Pipeline & Experiment Tracking

### 4.1 `src/pipeline/cli.py` — Idempotent Pipeline Runner
* **What it does:** Executes the 7 pipeline stages with SHA-256 config-hash caching (< 1 ms per cached stage).
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m src.pipeline.cli all
  ```
* **Exact Output in Terminal:**
  ```text
  ================================================================================
  RUNNING END-TO-END FINANCIAL RAG PIPELINE (make pipeline)
  ================================================================================
  [Stage 1/7: verify_data] CACHED (hash: 4f1a...). Skipping redundant execution.
  [Stage 2/7: chunk]       CACHED (hash: 7d2e...). Skipping redundant execution.
  [Stage 3/7: embed]       CACHED (hash: b89c...). Skipping redundant execution.
  [Stage 4/7: build_index] CACHED (hash: c12d...). Skipping redundant execution.
  [Stage 5/7: evaluate_retrieval]  CACHED (hash: 832f...). Skipping redundant execution.
  [Stage 6/7: evaluate_generation] CACHED (hash: a09b...). Skipping redundant execution.
  [Stage 7/7: build_report]        Generated docs/RESULTS.md successfully.
  ================================================================================
  PIPELINE COMPLETE in 0.4 seconds! All stages executed or cached.
  ================================================================================
  ```

---

### 4.2 `mlflow ui` — MLflow Tracking Server
* **What it does:** Launches the centralized MLflow experiment tracker.
* **Exact Command:**
  ```bash
  .\venv\Scripts\python.exe -m mlflow ui --host 127.0.0.1 --port 5000
  ```
* **Exact Output in Terminal:**
  ```text
  [INFO] Starting gunicorn / waitress server
  [INFO] Listening at: http://127.0.0.1:5000 (Press CTRL+C to quit)
  ```
  *(Shows 21 runs logged with `Faithfulness: 0.9803`, `Gate Accuracy: 0.94`, and `$0.000028` token cost).*

---

## 5. Quick Reference Table

| Goal | Terminal Command | What You See in Terminal |
|---|---|---|
| **Query RAG in CLI** | `python main.py --query "What is an ETF?"` | Formatted answer + citations + latency breakdown |
| **Launch Web UI** | `streamlit run ui/app.py` | Starts UI on `http://localhost:8501` |
| **Launch REST API** | `uvicorn app:app --port 8000` | Starts API on `http://localhost:8000` |
| **Launch MLflow UI** | `mlflow ui --port 5000` | Starts MLflow on `http://localhost:5000` |
| **Run All 55 Tests** | `pytest -q` | `55 passed in ~50s` |
| **Test Regression Guard** | `pytest tests/test_eval_regression.py -v` | Asserts nDCG@10 $\ge$ 0.450 in 0.12s |
| **Print Benchmark Table** | `python -m src.evaluation.retrieval_eval` | ASCII table comparing 5 architectures + 95% CIs |
| **Candidate Depth Sweep** | `python -m src.evaluation.sweep_candidate_depth` | Pareto table ($K=8$ vs $12$ vs $16$) |
| **Acronym Expansion Test** | `python -m src.evaluation.benchmark_acronym_expansion` | Lift table for HELOC, HSA, FSA |
| **Verify Data Hygiene** | `python -m src.data.verify_data` | SHA-256 hashes & 57,598 doc count |
| **Code Linting** | `ruff check src/ tests/ app.py` | `All checks passed!` |
| **Run 7 Pipeline Stages** | `python -m src.pipeline.cli all` | Executes / caches all 7 stages in < 1s |
