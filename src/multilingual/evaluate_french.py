"""Evaluation script for the French Multilingual RAG Engine.

Evaluates retrieval Hit@1, Hit@3, MRR, Gate accuracy, and Circuit Breaker
abstention across 20 French queries (16 factual shop policies + 4 out-of-domain traps).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.multilingual.french_engine import FrenchShopRAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_french_system(
    data_path: str = "data/french_shop_faq.json",
    eval_queries_path: str = "data/french_eval_queries.json",
    output_path: str = "results/french_evaluation_results.json",
    use_llm: bool = False,
) -> dict[str, Any]:
    """Execute evaluation of the French RAG engine."""
    eval_file = Path(eval_queries_path)
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation queries not found at {eval_file}")

    with open(eval_file, encoding="utf-8") as f:
        queries = json.load(f)

    logger.info(f"Initializing French RAG Engine with {data_path}...")
    engine = FrenchShopRAGEngine(data_path=data_path)

    query_results = []
    hit_1_count = 0
    hit_3_count = 0
    reciprocal_ranks = []
    answerable_gate_correct = 0
    unanswerable_gate_correct = 0
    latencies = []

    answerable_total = sum(1 for q in queries if q.get("is_answerable", True))
    unanswerable_total = sum(1 for q in queries if not q.get("is_answerable", True))

    logger.info(
        f"Evaluating {len(queries)} queries ({answerable_total} answerable, {unanswerable_total} unanswerable)..."
    )

    for q in queries:
        qid = q["query_id"]
        question = q["question"]
        is_answerable = q.get("is_answerable", True)
        target_doc = q.get("target_doc_id")

        # Run engine
        result = engine.run(question, top_k=3, use_llm=use_llm)
        latencies.append(result.latency_ms)

        retrieved_ids = [s["doc_id"] for s in result.sources]
        dense_sim = result.confidence_score

        # Retrieval metrics (evaluated on answerable queries where ground truth target exists)
        hit_1 = False
        hit_3 = False
        rr = 0.0

        if is_answerable and target_doc:
            if retrieved_ids and retrieved_ids[0] == target_doc:
                hit_1 = True
                hit_1_count += 1

            if target_doc in retrieved_ids[:3]:
                hit_3 = True
                hit_3_count += 1
                rank = retrieved_ids[:3].index(target_doc) + 1
                rr = 1.0 / rank

            reciprocal_ranks.append(rr)

            if not result.abstained:
                answerable_gate_correct += 1
        else:
            # For unanswerable queries, expected behavior is to ABSTAIN
            if result.abstained:
                unanswerable_gate_correct += 1

        query_results.append(
            {
                "query_id": qid,
                "question": question,
                "is_answerable": is_answerable,
                "target_doc_id": target_doc,
                "top_retrieved_id": retrieved_ids[0] if retrieved_ids else None,
                "top_3_retrieved_ids": retrieved_ids,
                "hit_at_1": hit_1 if is_answerable else None,
                "hit_at_3": hit_3 if is_answerable else None,
                "reciprocal_rank": rr if is_answerable else None,
                "confidence_score": dense_sim,
                "abstained": result.abstained,
                "gate_correct": (not result.abstained) if is_answerable else result.abstained,
                "answer_preview": result.answer[:140] + "..." if len(result.answer) > 140 else result.answer,
                "latency_ms": result.latency_ms,
            }
        )

    # Compute aggregate metrics
    hit_at_1_rate = round(hit_1_count / answerable_total, 4) if answerable_total else 0.0
    hit_at_3_rate = round(hit_3_count / answerable_total, 4) if answerable_total else 0.0
    mrr = round(sum(reciprocal_ranks) / answerable_total, 4) if answerable_total else 0.0
    answerable_gate_acc = round(answerable_gate_correct / answerable_total, 4) if answerable_total else 0.0
    circuit_breaker_acc = round(unanswerable_gate_correct / unanswerable_total, 4) if unanswerable_total else 0.0
    overall_gate_acc = round((answerable_gate_correct + unanswerable_gate_correct) / len(queries), 4)
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    summary = {
        "dataset": "French Boutique E-Commerce FAQ",
        "total_documents": len(engine.docs),
        "total_queries": len(queries),
        "answerable_queries": answerable_total,
        "unanswerable_traps": unanswerable_total,
        "embedding_model": engine.model_name,
        "retrieval_metrics": {
            "hit_at_1": hit_at_1_rate,
            "hit_at_3": hit_at_3_rate,
            "mean_reciprocal_rank_mrr": mrr,
        },
        "guardrail_metrics": {
            "answerable_gate_pass_rate": answerable_gate_acc,
            "unanswerable_circuit_breaker_accuracy": circuit_breaker_acc,
            "overall_gate_accuracy": overall_gate_acc,
        },
        "performance": {
            "avg_latency_ms": avg_latency,
            "min_latency_ms": round(min(latencies), 2) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        },
        "queries": query_results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"French evaluation results saved to {out_file}")
    print("\n" + "=" * 65)
    print(" [FR] FRENCH MULTILINGUAL RAG EVALUATION BENCHMARK [FR]")
    print("=" * 65)
    print(f"Total Policy Documents  : {len(engine.docs)}")
    print(f"Total Evaluation Queries: {len(queries)} (16 Factual + 4 Unanswerable Traps)")
    print(f"Embedding Architecture  : {engine.model_name}")
    print("-" * 65)
    print(f"Retrieval Hit@1         : {hit_at_1_rate * 100:.1f}% ({hit_1_count}/{answerable_total})")
    print(f"Retrieval Hit@3         : {hit_at_3_rate * 100:.1f}% ({hit_3_count}/{answerable_total})")
    print(f"Retrieval MRR           : {mrr:.4f}")
    print("-" * 65)
    print(f"Answerable Gate Pass    : {answerable_gate_acc * 100:.1f}% ({answerable_gate_correct}/{answerable_total})")
    print(f"Trap Circuit Breaker    : {circuit_breaker_acc * 100:.1f}% ({unanswerable_gate_correct}/{unanswerable_total})")
    print(f"Overall Gate Accuracy   : {overall_gate_acc * 100:.1f}%")
    print(f"Mean Query Latency      : {avg_latency:.2f} ms")
    print("=" * 65 + "\n")

    return summary


if __name__ == "__main__":
    evaluate_french_system()
