"""Tests for the French Multilingual RAG extension module."""

import json
from pathlib import Path

from src.multilingual.french_engine import FrenchShopRAGEngine, tokenize_french


def test_french_tokenization():
    text = "Délai de rétractation & remboursement sous 14 jours!"
    tokens = tokenize_french(text)
    assert "délai" in tokens
    assert "rétractation" in tokens
    assert "remboursement" in tokens
    assert "14" in tokens
    assert "&" not in tokens


def test_french_faq_dataset_structure():
    data_path = Path("data/french_shop_faq.json")
    assert data_path.exists(), "french_shop_faq.json must exist"

    with open(data_path, encoding="utf-8") as f:
        docs = json.load(f)

    assert len(docs) == 30, f"Expected exactly 30 French FAQ documents, got {len(docs)}"
    doc_ids = set()
    for d in docs:
        assert "doc_id" in d and d["doc_id"].startswith("fr_doc_")
        assert "category" in d and len(d["category"]) > 0
        assert "title" in d and len(d["title"]) > 0
        assert "content" in d and len(d["content"]) >= 30
        assert d["doc_id"] not in doc_ids, f"Duplicate doc_id: {d['doc_id']}"
        doc_ids.add(d["doc_id"])


def test_french_eval_queries_structure():
    eval_path = Path("data/french_eval_queries.json")
    assert eval_path.exists(), "french_eval_queries.json must exist"

    with open(eval_path, encoding="utf-8") as f:
        queries = json.load(f)

    assert len(queries) == 20, f"Expected exactly 20 French eval queries, got {len(queries)}"
    answerable = [q for q in queries if q.get("is_answerable", True)]
    unanswerable = [q for q in queries if not q.get("is_answerable", True)]

    assert len(answerable) == 16, f"Expected 16 answerable queries, got {len(answerable)}"
    assert len(unanswerable) == 4, f"Expected 4 unanswerable traps, got {len(unanswerable)}"

    # Check target doc IDs exist in corpus
    faq_path = Path("data/french_shop_faq.json")
    with open(faq_path, encoding="utf-8") as f:
        faq_ids = {d["doc_id"] for d in json.load(f)}

    for q in answerable:
        assert q["target_doc_id"] in faq_ids, f"Target {q['target_doc_id']} not found in FAQ corpus"


def test_french_engine_retrieval_and_answer():
    engine = FrenchShopRAGEngine()
    query = "Quel est le délai de rétractation et de retour d'un colis ?"
    result = engine.run(query, top_k=3, use_llm=False)

    assert not result.abstained
    assert result.sources[0]["doc_id"] == "fr_doc_01"
    assert "14 jours" in result.answer
    assert "[Source 1]" in result.answer
    assert result.latency_ms > 0


def test_french_circuit_breaker_abstention():
    engine = FrenchShopRAGEngine()
    out_of_domain = "Livrez-vous des fusées spatiales à destination de la planète Mars ?"
    result = engine.run(out_of_domain, top_k=3, use_llm=False)

    assert result.abstained is True
    assert "Désolé, aucune information officielle" in result.answer
    assert result.confidence_score < engine.abstain_threshold


def test_french_evaluation_results_artifact():
    results_path = Path("results/french_evaluation_results.json")
    assert results_path.exists(), "french_evaluation_results.json must exist"

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_documents"] == 30
    assert data["total_queries"] == 20
    assert data["retrieval_metrics"]["hit_at_1"] >= 0.80
    assert data["retrieval_metrics"]["hit_at_3"] >= 0.90
    assert data["retrieval_metrics"]["mean_reciprocal_rank_mrr"] >= 0.85
    assert data["guardrail_metrics"]["overall_gate_accuracy"] >= 0.90


def test_evaluate_french_runner(tmp_path):
    from src.multilingual.evaluate_french import evaluate_french_system

    mini_queries = [
        {
            "query_id": "MINI_01",
            "question": "Quel est le délai légal pour retourner un article ?",
            "target_doc_id": "fr_doc_01",
            "is_answerable": True,
        },
        {
            "query_id": "MINI_02",
            "question": "Faites-vous la réparation de tondeuses ?",
            "is_answerable": False,
        },
    ]
    query_file = tmp_path / "mini_queries.json"
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(mini_queries, f)

    out_file = tmp_path / "out_report.json"
    summary = evaluate_french_system(
        eval_queries_path=str(query_file),
        output_path=str(out_file),
        use_llm=False,
    )

    assert summary["total_queries"] == 2
    assert summary["answerable_queries"] == 1
    assert summary["unanswerable_traps"] == 1
    assert out_file.exists()
