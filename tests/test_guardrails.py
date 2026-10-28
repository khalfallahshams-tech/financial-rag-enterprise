from src.guardrails.confidence_gate import ConfidenceGate
from src.guardrails.citation_auditor import CitationAuditor
from src.schemas import GateStatus

def test_confidence_gate_passes_high_score():
    gate = ConfidenceGate(threshold=0.0)
    candidates = [{"score": 3.45, "text": "Relevant context"}]
    status, score, refusal = gate.evaluate(candidates)

    assert status == GateStatus.PASSED
    assert score == 3.45
    assert refusal == ""

def test_confidence_gate_blocks_negative_score():
    gate = ConfidenceGate(threshold=0.0)
    candidates = [{"score": -4.21, "text": "Irrelevant text"}]
    status, score, refusal = gate.evaluate(candidates)

    assert status == GateStatus.BLOCKED
    assert score == -4.21
    assert "does not contain sufficient information" in refusal

def test_confidence_gate_blocks_empty_candidates():
    gate = ConfidenceGate(threshold=0.0)
    status, score, refusal = gate.evaluate([])

    assert status == GateStatus.BLOCKED

def test_citation_auditor_perfect_precision():
    sources = [{"source_tag": "Source 1", "chunk_id": "doc_6301_c0"}]
    text = "Endorse the check on the back [Source 1: doc_6301_c0]."
    report = CitationAuditor.audit(text, sources)

    assert report.total_citations == 1
    assert report.valid_citations == 1
    assert report.hallucinated_citations == 0
    assert report.citation_precision == 1.0

def test_citation_auditor_flags_fake_citation():
    sources = [{"source_tag": "Source 1", "chunk_id": "doc_6301_c0"}]
    text = "A fake claim backed by a fake source [Source 99: doc_99999_c0]."
    report = CitationAuditor.audit(text, sources)

    assert report.total_citations == 1
    assert report.valid_citations == 0
    assert report.hallucinated_citations == 1
    assert report.citation_precision == 0.0

