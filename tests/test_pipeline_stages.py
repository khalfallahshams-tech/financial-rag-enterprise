from pathlib import Path
from src.pipeline.stages import compute_stage_hash, stage_verify_data, stage_chunk, stage_embed, stage_build_index
from src.schemas import QueryRequest, GateStatus
from src.pipeline import FinancialRAGPipeline


def test_stage_hash_determinism(tmp_path):
    f1 = tmp_path / "test1.txt"
    f1.write_text("sample content")
    
    cfg = {"param": 42}
    h1 = compute_stage_hash([f1], cfg)
    h2 = compute_stage_hash([f1], cfg)
    assert h1 == h2

    # Invalidate by changing config
    h3 = compute_stage_hash([f1], {"param": 99})
    assert h1 != h3


def test_stage_verify_data_structure():
    report = stage_verify_data(force=False)
    assert "corpus" in report
    assert "academic_eval" in report
    assert "enterprise_stress_eval" in report
    assert report["corpus"]["unique_valid_docs"] == 57598


def test_pipeline_stages_idempotency():
    # Calling stage_chunk twice should succeed and be consistent
    r1 = stage_chunk(force=False)
    r2 = stage_chunk(force=False)
    assert r1["total_indexed_chunks"] == r2["total_indexed_chunks"]
    assert r1["status"] == "VALID"


def test_pipeline_orchestrator_execution():
    pipeline = FinancialRAGPipeline()
    req = QueryRequest(question="What is an emergency fund?", use_cache=True)
    resp = pipeline.run(req)

    assert resp.question == "What is an emergency fund?"
    assert resp.gate_decision in (GateStatus.PASSED, GateStatus.BLOCKED)
    assert len(resp.citations) <= 3
    assert resp.total_latency_ms > 0
