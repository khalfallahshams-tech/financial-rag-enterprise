import numpy as np
import pytest

from src.evaluation.metrics import (
    bootstrap_confidence_interval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_ndcg_perfect_ranking():
    # When retrieved items are ordered by relevance, nDCG must be exactly 1.0
    rels = [3, 2, 1, 0, 0]
    score = ndcg_at_k(rels, k=5)
    assert pytest.approx(score, rel=1e-5) == 1.0


def test_ndcg_zero_relevance():
    # When no items are relevant, nDCG must be 0.0
    rels = [0, 0, 0]
    score = ndcg_at_k(rels, k=3)
    assert score == 0.0


def test_ndcg_hand_computed():
    # Binary relevance: retrieved [0, 1] vs ideal [1, 0]
    # Actual DCG@2: (2^0 - 1)/log2(2) + (2^1 - 1)/log2(3) = 0 + 1 / 1.58496 = 0.63093
    # Ideal DCG@2: (2^1 - 1)/log2(2) + (2^0 - 1)/log2(3) = 1.0
    # Expected nDCG@2: ~0.63093
    rels = [0, 1]
    score = ndcg_at_k(rels, k=2)
    expected = 1.0 / np.log2(3)
    assert pytest.approx(score, rel=1e-4) == expected


def test_recall_at_k():
    ground_truth = ["doc_A", "doc_B", "doc_C"]
    retrieved = ["doc_A", "doc_D", "doc_B", "doc_E"]

    # In top 2: only doc_A found -> 1/3
    assert pytest.approx(recall_at_k(retrieved, ground_truth, k=2), rel=1e-4) == 1.0 / 3.0
    # In top 3: doc_A and doc_B found -> 2/3
    assert pytest.approx(recall_at_k(retrieved, ground_truth, k=3), rel=1e-4) == 2.0 / 3.0
    # In top 4: doc_A and doc_B found -> 2/3
    assert pytest.approx(recall_at_k(retrieved, ground_truth, k=4), rel=1e-4) == 2.0 / 3.0


def test_reciprocal_rank():
    ground_truth = ["doc_target"]
    assert reciprocal_rank(["doc_target", "doc_other"], ground_truth) == 1.0
    assert reciprocal_rank(["doc_other", "doc_target"], ground_truth) == 0.5
    assert reciprocal_rank(["doc_other1", "doc_other2"], ground_truth) == 0.0


def test_bootstrap_ci_properties():
    data = [0.2, 0.4, 0.6, 0.8, 1.0]
    mean, lower, upper = bootstrap_confidence_interval(data, n_bootstraps=500, ci=0.95, seed=42)

    assert pytest.approx(mean, rel=1e-5) == 0.6
    assert lower <= mean <= upper
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0

    # Determinism test
    mean2, lower2, upper2 = bootstrap_confidence_interval(data, n_bootstraps=500, ci=0.95, seed=42)
    assert mean == mean2
    assert lower == lower2
    assert upper == upper2

