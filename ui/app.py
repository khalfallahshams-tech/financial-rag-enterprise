import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import httpx
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="FiQA Financial RAG Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Default API URL
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

# -----------------------------------------------------------------------------
# API Client Helpers
# -----------------------------------------------------------------------------
def fetch_health(api_url: str) -> Dict[str, Any]:
    try:
        r = httpx.get(f"{api_url}/health", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def fetch_config(api_url: str) -> Dict[str, Any]:
    try:
        r = httpx.get(f"{api_url}/config", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def send_query(api_url: str, question: str, retriever: str, top_k: int, use_cache: bool) -> Dict[str, Any]:
    payload = {
        "question": question,
        "retriever": retriever,
        "top_k": top_k,
        "use_cache": use_cache
    }
    r = httpx.post(f"{api_url}/query", json=payload, timeout=35.0)
    if r.status_code == 200:
        return r.json()
    elif r.status_code == 422:
        return {"error": f"Validation Error (422): {r.text}"}
    elif r.status_code == 504:
        return {"error": "Gateway Timeout (504): The request exceeded the 30-second pipeline SLA."}
    else:
        return {"error": f"API Error ({r.status_code}): {r.text}"}

def clear_api_cache(api_url: str) -> bool:
    try:
        r = httpx.delete(f"{api_url}/cache", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Sidebar: Telemetry & Controls
# -----------------------------------------------------------------------------
st.sidebar.title("🏦 FiQA RAG Engine")
api_base = st.sidebar.text_input("Backend API Base URL", value=DEFAULT_API_URL)

health_info = fetch_health(api_base)
config_info = fetch_config(api_base)

if health_info and health_info.get("status") == "healthy":
    st.sidebar.success(f"🟢 Backend Online (ChromaDB: {health_info.get('chroma_db_status')})")
    st.sidebar.caption(f"📚 Indexed Chunks: **{health_info.get('indexed_chunks_count', 0):,}**")
    st.sidebar.caption(f"💾 In-Memory Cache: **{health_info.get('cache_entries_count', 0)} entries**")
else:
    st.sidebar.error("🔴 Backend Offline or Unreachable")
    st.sidebar.caption("Run `uvicorn app:app --port 8000` to start the backend.")

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval Strategy")
retriever_option = st.sidebar.radio(
    "Choose Strategy:",
    options=["hybrid_rerank", "hybrid", "dense", "bm25"],
    format_func=lambda x: {
        "hybrid_rerank": "🌟 Hybrid + Re-Ranker (Champion)",
        "hybrid": "⚡ Hybrid RRF (Dense + BM25)",
        "dense": "🔍 Pure Dense (MiniLM / BGE)",
        "bm25": "📖 Pure Sparse (BM25Okapi)"
    }[x],
    index=0
)

top_k_val = st.sidebar.slider("Top-K Sources to Return", min_value=1, max_value=8, value=3)
use_cache_val = st.sidebar.checkbox("Enable In-Memory Cache", value=True)

if st.sidebar.button("🧹 Clear In-Memory Cache"):
    if clear_api_cache(api_base):
        st.sidebar.info("Cache cleared successfully!")
    else:
        st.sidebar.warning("Could not clear cache (backend offline).")

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
st.title("🏦 Enterprise Financial RAG Intelligence")
st.markdown(
    "**Two-Stage Hybrid Architecture**: BM25 exact keyword matching + Dense semantic vector search, "
    "fused into a **Cross-Encoder neural re-ranker** with **tau=0.0 circuit breaker guardrails** and citation attribution."
)

tab_query, tab_compare, tab_eval = st.tabs([
    "💬 Live Financial Query",
    "⚖️ Compare Retrievers Side-by-Side",
    "📊 Evaluation & Academic Benchmarks"
])

# -----------------------------------------------------------------------------
# TAB 1: Live Financial Query
# -----------------------------------------------------------------------------
with tab_query:
    st.subheader("Ask a Financial Compliance or Investment Question")

    # Sample queries pills
    sample_queries = [
        "What is an ETF and how does it trade on the exchange?",
        "Can I roll over a traditional 401(k) to a Roth IRA?",
        "How do interest rates and equity drawdowns work on a HELOC?",
        "What is the capital gains tax rate on primary home sales?",
        "What are supersymmetric multiversal string quantum states? (OOD Trap)"
    ]

    selected_sample = st.selectbox(
        "💡 Quick-fill example questions:",
        options=["-- Type custom question below --"] + sample_queries
    )

    default_text = "" if selected_sample.startswith("--") else selected_sample
    user_query = st.text_area(
        "Enter Financial Query:",
        value=default_text,
        placeholder="e.g., How does an investor calculate compound interest for retirement?",
        height=90
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("🚀 Run RAG Query", type="primary", use_container_width=True)

    if run_btn:
        if not user_query.strip():
            st.warning("Please enter a valid financial question.")
        else:
            with st.spinner("Executing Two-Stage Hybrid Retrieval, Re-ranking, and Defensive Generation..."):
                t0 = time.time()
                res = send_query(api_base, user_query.strip(), retriever_option, top_k_val, use_cache_val)
                elapsed_ms = round((time.time() - t0) * 1000, 1)

            if "error" in res:
                st.error(res["error"])
            else:
                st.markdown("---")
                # Top Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Latency", f"{res.get('latency_ms', elapsed_ms):.1f} ms")
                with m2:
                    gate_status = res.get("gate_decision", "UNKNOWN")
                    is_abstained = res.get("abstained", False)
                    if is_abstained or "BLOCKED" in gate_status:
                        st.metric("Guardrail Gate", "🛑 BLOCKED", delta="0 LLM Tokens", delta_color="inverse")
                    else:
                        st.metric("Guardrail Gate", "🟢 PASSED", delta="Evidence Verified")
                with m3:
                    is_cached = res.get("is_cached", False)
                    st.metric("Cache Status", "⚡ CACHE HIT" if is_cached else "🔄 FRESH RUN")
                with m4:
                    score = res.get("confidence_score", 0.0)
                    st.metric("Confidence Score", f"{score:.3f}")

                # Abstention Banner vs Verified Answer
                if res.get("abstained", False):
                    st.warning(
                        f"**Circuit Breaker Activated (Low Relevance / Out-of-Domain Guardrail)**\n\n"
                        f"{res.get('answer')}\n\n"
                        f"*Notice: Zero external LLM calls were made. This halts hallucination traps and protects token budget.*"
                    )
                else:
                    st.markdown("### 📝 Verified Answer")
                    st.info(res.get("answer", ""))

                # Expanded Acronyms Banner if applicable
                acronyms = res.get("expanded_acronyms", [])
                if acronyms:
                    st.caption(f"🔍 **Domain Acronyms Expanded:** `{', '.join(acronyms)}`")

                # Latency Decomposition
                with st.expander("⏱️ Latency Decomposition Breakdown", expanded=False):
                    l_cols = st.columns(4)
                    l_cols[0].metric("Retrieval (Dense+BM25)", f"{res.get('retrieval_latency_ms', 0):.1f} ms")
                    l_cols[1].metric("Cross-Encoder Rerank", f"{res.get('rerank_latency_ms', 0):.1f} ms")
                    l_cols[2].metric("Defensive LLM Generation", f"{res.get('llm_latency_ms', 0):.1f} ms")
                    l_cols[3].metric("Total End-to-End", f"{res.get('total_latency_ms', 0):.1f} ms")

                # Citation Audit
                audit = res.get("citation_audit", {})
                if audit:
                    with st.expander("🛡️ Citation Attribution Audit", expanded=False):
                        a_cols = st.columns(3)
                        a_cols[0].metric("Citation Precision", f"{audit.get('citation_precision', 1.0)*100:.1f}%")
                        a_cols[1].metric("Valid Citations", f"{audit.get('valid_citations', 0)} / {audit.get('total_citations', 0)}")
                        a_cols[2].metric("Sentence Coverage", f"{audit.get('sentence_coverage', 1.0)*100:.1f}%")

                # Retrieved Passages / Citations
                st.markdown("### 📚 Grounding Source Passages")
                sources = res.get("sources", [])
                if not sources and res.get("citations"):
                    # Fallback to citations if sources empty
                    for cit in res.get("citations", []):
                        with st.container():
                            st.markdown(f"**{cit['source_tag']}** — `{cit['chunk_id']}` (Cross-Encoder Score: `{cit['score']:.2f}`)")
                            st.write(cit["snippet"])
                else:
                    for idx, src in enumerate(sources, 1):
                        with st.container():
                            st.markdown(f"**Source {idx}** — Chunk ID: `{src['id']}` | Score: `{src['score']:.2f}`")
                            st.write(src["text"])

# -----------------------------------------------------------------------------
# TAB 2: Compare Retrievers Side-by-Side
# -----------------------------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ Side-by-Side Retriever Benchmark Comparison")
    st.caption("Compare how Pure BM25, Pure Dense, Hybrid RRF, and Champion Hybrid + Cross-Encoder handle the same query.")

    comp_query = st.text_input(
        "Comparison Query:",
        value="What is a HELOC and how are its payments structured?",
        key="comp_query_input"
    )

    if st.button("⚡ Compare All 4 Retrievers", type="secondary"):
        if not comp_query.strip():
            st.warning("Please enter a query.")
        else:
            strategies = [
                ("bm25", "📖 BM25 (Lexical)"),
                ("dense", "🔍 Dense (MiniLM)"),
                ("hybrid", "⚡ Hybrid RRF"),
                ("hybrid_rerank", "🌟 Hybrid + Re-Ranker")
            ]

            results_comp = []
            progress_bar = st.progress(0.0)

            for idx, (strat_code, strat_label) in enumerate(strategies):
                t0 = time.time()
                res = send_query(api_base, comp_query.strip(), strat_code, top_k=2, use_cache=False)
                latency = round((time.time() - t0) * 1000, 1)

                if "error" in res:
                    results_comp.append({
                        "Retriever": strat_label,
                        "Latency (ms)": latency,
                        "Abstained": True,
                        "Top Score": 0.0,
                        "Top Source": "N/A",
                        "Answer Preview": res["error"][:100]
                    })
                else:
                    top_src = res["sources"][0]["text"][:120] + "..." if res.get("sources") else "No source"
                    results_comp.append({
                        "Retriever": strat_label,
                        "Latency (ms)": res.get("latency_ms", latency),
                        "Abstained": res.get("abstained", False),
                        "Top Score": round(res.get("confidence_score", 0.0), 3),
                        "Top Source": top_src,
                        "Answer Preview": res.get("answer", "")[:140] + "..."
                    })
                progress_bar.progress((idx + 1) / len(strategies))

            df_comp = pd.DataFrame(results_comp)
            st.dataframe(df_comp, use_container_width=True)

            # Chart comparing latencies
            st.bar_chart(df_comp.set_index("Retriever")["Latency (ms)"])

# -----------------------------------------------------------------------------
# TAB 3: Academic Benchmarks & Evaluation Reports
# -----------------------------------------------------------------------------
with tab_eval:
    st.subheader("📊 Rigorous Evaluation & Academic Benchmarks")
    st.caption("All numbers are empirically reproduced on the BEIR FiQA test set with 95% non-parametric bootstrap confidence intervals.")

    # 1. Retrieval Benchmark Table
    retrieval_json_path = RESULTS_DIR / "beir_retrieval_benchmark_results.json"
    if retrieval_json_path.exists():
        with open(retrieval_json_path, "r", encoding="utf-8") as f:
            ret_data = json.load(f)

        st.markdown("#### 1. BEIR FiQA Retrieval Benchmark (30 Academic Queries)")
        table_rows = []
        for name, metrics in ret_data.items():
            ndcg = metrics.get("ndcg_at_10", {})
            rec10 = metrics.get("recall_at_10", {})
            rec100 = metrics.get("recall_at_100", {})
            mrr = metrics.get("mrr", {})
            table_rows.append({
                "Architecture": name,
                "nDCG@10 (95% CI)": f"{ndcg.get('mean', 0):.3f} [{ndcg.get('ci_low', 0):.3f}, {ndcg.get('ci_high', 0):.3f}]",
                "Recall@10 (95% CI)": f"{rec10.get('mean', 0)*100:.1f}% [{rec10.get('ci_low', 0)*100:.1f}%, {rec10.get('ci_high', 0)*100:.1f}%]",
                "Recall@100 (95% CI)": f"{rec100.get('mean', 0)*100:.1f}% [{rec100.get('ci_low', 0)*100:.1f}%, {rec100.get('ci_high', 0)*100:.1f}%]",
                "MRR": f"{mrr.get('mean', 0):.3f}",
                "P95 Latency": f"{metrics.get('latency_p95_ms', 0):.0f} ms"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # 2. Generation & Guardrail Benchmark
    gen_json_path = RESULTS_DIR / "generation_evaluation_results.json"
    if gen_json_path.exists():
        with open(gen_json_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)

        st.markdown("#### 2. Generation & Guardrail Evaluation (50-Item Golden Set)")
        g1, g2, g3, g4 = st.columns(4)
        faith = gen_data.get("faithfulness", {})
        relev = gen_data.get("answer_relevancy", {})
        gate = gen_data.get("gate_accuracy", {})
        cost = gen_data.get("cost_summary", {})

        g1.metric("Faithfulness (95% CI)", f"{faith.get('mean', 0)*100:.1f}%", f"[{faith.get('ci_low', 0)*100:.1f}%, {faith.get('ci_high', 0)*100:.1f}%]")
        g2.metric("Answer Relevancy (95% CI)", f"{relev.get('mean', 0)*100:.1f}%", f"[{relev.get('ci_low', 0)*100:.1f}%, {relev.get('ci_high', 0)*100:.1f}%]")
        g3.metric("Gate Accuracy (95% CI)", f"{gate.get('mean', 0)*100:.1f}%", f"[{gate.get('ci_low', 0)*100:.1f}%, {gate.get('ci_high', 0)*100:.1f}%]")
        g4.metric("Avg Cost / Query", f"${cost.get('cost_per_query_usd', 0):.6f}", f"${cost.get('total_cost_usd', 0):.4f} total")

    # 3. Candidate Depth Sweep & Acronym Benchmark
    col_sw, col_ac = st.columns(2)
    with col_sw:
        sweep_path = RESULTS_DIR / "candidate_depth_sweep.json"
        if sweep_path.exists():
            st.markdown("#### 3. Candidate Pool Sweep ($K \\in [8, 12, 16]$)")
            with open(sweep_path, "r", encoding="utf-8") as f:
                sw_data = json.load(f)
            sw_rows = []
            for k_str, m in sw_data.items():
                sw_rows.append({
                    "Candidate Pool": f"K={k_str}",
                    "Hit@1": f"{m.get('hit_at_1', 0)*100:.1f}%",
                    "Hit@3": f"{m.get('hit_at_3', 0)*100:.1f}%",
                    "MRR": f"{m.get('mrr', 0):.3f}",
                    "Latency (ms)": f"{m.get('latency_mean_ms', 0):.0f} ms"
                })
            st.dataframe(pd.DataFrame(sw_rows), use_container_width=True)

    with col_ac:
        ac_path = RESULTS_DIR / "acronym_expansion_benchmark.json"
        if ac_path.exists():
            st.markdown("#### 4. Acronym Query Expansion Impact")
            with open(ac_path, "r", encoding="utf-8") as f:
                ac_data = json.load(f)
            ac_rows = []
            for query_text, item in ac_data.get("individual_queries", {}).items():
                ac_rows.append({
                    "Query Acronym": query_text[:35] + "...",
                    "Raw Score": f"{item.get('raw_score', 0):.2f}",
                    "Expanded Score": f"{item.get('expanded_score', 0):.2f}",
                    "Delta": f"{item.get('score_delta', 0):+.2f}"
                })
            st.dataframe(pd.DataFrame(ac_rows[:6]), use_container_width=True)

    # 4. Full Executive Report
    rep_path = RESULTS_DIR / "executive_evaluation_report.md"
    if rep_path.exists():
        with st.expander("📄 View Full Executive Evaluation Report Markdown", expanded=False):
            with open(rep_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
