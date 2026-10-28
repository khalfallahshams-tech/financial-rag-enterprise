"""
Data integrity and hygiene verification module for FiQA RAG dataset.
Performs deterministic audits of corpus files, evaluation suites, and pre-computed indexes.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import chromadb
import pandas as pd


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 16):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_corpus(corpus_path: Path) -> Dict[str, Any]:
    """Verify raw corpus CSV integrity, nulls, duplicates, and token length profile."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}")

    file_size_mb = corpus_path.stat().st_size / (1024 * 1024)
    file_hash = compute_file_sha256(corpus_path)

    df = pd.read_csv(corpus_path)
    total_rows = len(df)
    null_rows = int(df["doc"].isnull().sum())
    
    # Cleaned rows
    df_clean = df.dropna(subset=["doc"])
    duplicate_rows = int(df_clean.duplicated(subset=["doc"]).sum())
    unique_rows = int(df_clean["doc"].nunique())

    # Text length metrics
    char_lens = df_clean["doc"].str.len()
    word_lens = df_clean["doc"].str.split().str.len()

    return {
        "file_path": str(corpus_path),
        "file_size_mb": round(file_size_mb, 2),
        "sha256": file_hash,
        "total_rows": total_rows,
        "null_rows": null_rows,
        "duplicate_rows": duplicate_rows,
        "unique_valid_docs": unique_rows,
        "avg_char_length": round(float(char_lens.mean()), 1),
        "median_char_length": round(float(char_lens.median()), 1),
        "max_char_length": int(char_lens.max()),
        "min_char_length": int(char_lens.min()),
        "avg_word_count": round(float(word_lens.mean()), 1),
    }


def verify_academic_eval(eval_path: Path) -> Dict[str, Any]:
    """Verify Suite 1: Academic 30-query evaluation dataset (fiqa_eval.csv)."""
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation benchmark file not found at {eval_path}")

    file_size_kb = eval_path.stat().st_size / 1024
    file_hash = compute_file_sha256(eval_path)

    df = pd.read_csv(eval_path)
    required_cols = ["question", "ground_truths", "answer", "contexts"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {eval_path}: {missing_cols}")

    null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}
    total_queries = len(df)
    unique_queries = int(df["question"].nunique())

    return {
        "file_path": str(eval_path),
        "file_size_kb": round(file_size_kb, 2),
        "sha256": file_hash,
        "total_queries": total_queries,
        "unique_queries": unique_queries,
        "null_counts": null_counts,
        "columns": df.columns.tolist(),
    }


def verify_enterprise_stress_eval(eval_json_path: Path) -> Dict[str, Any]:
    """Verify Suite 2: Enterprise 45-query multi-category stress test (fiqa_eval_new.json)."""
    if not eval_json_path.exists():
        raise FileNotFoundError(f"Stress test JSON not found at {eval_json_path}")

    file_size_kb = eval_json_path.stat().st_size / 1024
    file_hash = compute_file_sha256(eval_json_path)

    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list of queries in {eval_json_path}")

    categories: Dict[str, int] = {}
    gate_decisions: Dict[str, int] = {}
    answerable_count = 0

    for item in data:
        cat = item.get("category", "UNKNOWN")
        categories[cat] = categories.get(cat, 0) + 1

        gate = item.get("expected_gate_decision", "UNKNOWN")
        gate_decisions[gate] = gate_decisions.get(gate, 0) + 1

        if item.get("is_answerable", True):
            answerable_count += 1

    return {
        "file_path": str(eval_json_path),
        "file_size_kb": round(file_size_kb, 2),
        "sha256": file_hash,
        "total_queries": len(data),
        "category_breakdown": categories,
        "expected_gate_decisions": gate_decisions,
        "answerable_count": answerable_count,
        "unanswerable_count": len(data) - answerable_count,
    }


def verify_chromadb(chroma_dir: Path) -> Dict[str, Any]:
    """Verify ChromaDB collections and vector counts."""
    if not chroma_dir.exists():
        raise FileNotFoundError(f"ChromaDB directory not found at {chroma_dir}")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collections = client.list_collections()
    col_info = {}
    for col in collections:
        col_info[col.name] = {
            "count": col.count(),
            "metadata": col.metadata,
        }

    return {
        "chroma_dir": str(chroma_dir),
        "collections": col_info,
    }


def verify_bm25_index(bm25_path: Path) -> Dict[str, Any]:
    """Verify BM25 precomputed pickle index."""
    if not bm25_path.exists():
        return {"status": "MISSING", "path": str(bm25_path)}

    file_size_mb = bm25_path.stat().st_size / (1024 * 1024)
    with open(bm25_path, "rb") as f:
        bm25_obj = pickle.load(f)

    doc_count = getattr(bm25_obj, "corpus_size", None)
    return {
        "status": "VALID",
        "file_path": str(bm25_path),
        "file_size_mb": round(file_size_mb, 2),
        "corpus_size": doc_count,
    }


def run_full_data_audit() -> Dict[str, Any]:
    """Execute complete data audit and return structured summary."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    chroma_dir = base_dir / "data" / "chroma_db"
    bm25_path = base_dir / "data" / "bm25_index.pkl"

    corpus_report = verify_corpus(raw_dir / "fiqa_corpus.csv")
    academic_eval_report = verify_academic_eval(raw_dir / "fiqa_eval.csv")
    stress_eval_report = verify_enterprise_stress_eval(raw_dir / "fiqa_eval_new.json")
    chroma_report = verify_chromadb(chroma_dir)
    bm25_report = verify_bm25_index(bm25_path)

    return {
        "corpus": corpus_report,
        "academic_eval": academic_eval_report,
        "enterprise_stress_eval": stress_eval_report,
        "chromadb": chroma_report,
        "bm25_index": bm25_report,
    }


if __name__ == "__main__":
    import pprint
    print("=" * 60)
    print("RUNNING DETERMINISTIC DATA AUDIT & HYGIENE VERIFICATION")
    print("=" * 60)
    report = run_full_data_audit()
    pprint.pprint(report)
    print("=" * 60)
    print("AUDIT COMPLETE: All data artifacts verified.")

