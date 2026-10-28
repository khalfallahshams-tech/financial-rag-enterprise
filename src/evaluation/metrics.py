"""
Information Retrieval (IR) and RAG Evaluation Metrics.
Implements:
- nDCG@K (Normalized Discounted Cumulative Gain)
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Non-parametric Bootstrap Confidence Intervals (95% CI)
"""

from collections.abc import Sequence
from typing import Any

import numpy as np


def dcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """
    Computes Discounted Cumulative Gain at rank K.
    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    rels = np.asarray(relevances, dtype=float)[:k]
    if rels.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0 ** rels - 1.0) / discounts))


def ndcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K.
    nDCG@K = DCG@K / IDCG@K
    """
    actual_dcg = dcg_at_k(relevances, k=k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k=k)
    if ideal_dcg == 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def recall_at_k(retrieved: Sequence[Any], ground_truth: Sequence[Any], k: int = 10) -> float:
    """
    Computes Recall@K: fraction of relevant ground-truth items found in top-K.
    """
    if not ground_truth:
        return 0.0
    gt_set = set(ground_truth)
    top_k_set = set(retrieved[:k])
    intersection = len(gt_set.intersection(top_k_set))
    return float(intersection / len(gt_set))


def reciprocal_rank(retrieved: Sequence[Any], ground_truth: Sequence[Any]) -> float:
    """
    Computes Reciprocal Rank (RR): 1 / rank of the first relevant document found.
    Returns 0.0 if no ground-truth document is retrieved.
    """
    gt_set = set(ground_truth)
    for rank, item in enumerate(retrieved, start=1):
        if item in gt_set:
            return 1.0 / rank
    return 0.0


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Calculates the mean and non-parametric bootstrap confidence interval for a metric sample.
    
    Returns:
        (mean, ci_lower, ci_upper)
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    if arr.size == 1:
        val = float(arr[0])
        return val, val, val

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_bootstraps, arr.size))
    boot_means = np.mean(arr[indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    point_mean = float(np.mean(arr))

    return point_mean, lower, upper
