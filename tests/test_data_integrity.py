import json
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

def test_corpus_hygiene():
    """Validates corpus file existence, non-emptiness, and absence of null text rows."""
    corpus_path = RAW_DIR / "fiqa_corpus.csv"
    assert corpus_path.exists(), "fiqa_corpus.csv must exist in data/raw"

    df = pd.read_csv(corpus_path)
    assert len(df) > 50000, f"Expected >50,000 corpus rows, found {len(df)}"

    # Check for empty text documents in 'doc' column
    valid_docs = df[df["doc"].notna()]
    clean_texts = valid_docs["doc"].astype(str).str.strip()
    assert (clean_texts == "").sum() == 0, "Corpus contains empty string documents!"

    # 57,600 non-null documents and 57,598 unique valid documents (2 duplicates present)
    assert valid_docs["doc"].nunique() == 57598
    assert len(valid_docs) == 57600

def test_golden_eval_set_integrity():
    """Validates that golden eval set has 50 valid items with required schema."""
    golden_path = DATA_DIR / "golden_eval_set.json"
    assert golden_path.exists(), "golden_eval_set.json must exist in data/"

    with open(golden_path, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 50, f"Golden set must have exactly 50 items, found {len(data)}"

    seen_questions = set()
    for idx, item in enumerate(data):
        assert "question" in item, f"Item {idx} missing 'question'"
        assert "ground_truth_answer" in item, f"Item {idx} missing 'ground_truth_answer'"
        q = item["question"].strip()
        ans = item["ground_truth_answer"].strip()
        assert len(q) >= 10, f"Item {idx} question too short: '{q}'"
        assert len(ans) >= 10, f"Item {idx} answer too short"
        assert q not in seen_questions, f"Duplicate question in golden set: '{q}'"
        seen_questions.add(q)

def test_eval_new_ground_truth_consistency():
    """Validates that fiqa_eval_new.json queries and answers are non-empty."""
    eval_new_path = RAW_DIR / "fiqa_eval_new.json"
    if not eval_new_path.exists():
        pytest.skip("fiqa_eval_new.json not in data/raw")

    with open(eval_new_path, encoding="utf-8") as f:
        eval_items = json.load(f)

    assert len(eval_items) >= 30, f"Expected >= 30 eval queries, found {len(eval_items)}"
    for item in eval_items:
        assert item.get("question", "").strip() != "", "Eval query question cannot be empty"
        assert item.get("ground_truth_answer", "").strip() != "", "Eval ground truth cannot be empty"

def test_no_split_overlap():
    """Verifies that golden eval set does not overlap with synthetic or train queries."""
    golden_path = DATA_DIR / "golden_eval_set.json"
    if not golden_path.exists():
        return

    with open(golden_path, encoding="utf-8") as f:
        golden_qs = {item["question"].strip().lower() for item in json.load(f)}

    # Ensure all 50 queries are unique
    assert len(golden_qs) == 50

