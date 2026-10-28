import pytest
from src.data.chunker import chunk_text
from src.retrieval.hybrid import reciprocal_rank_fusion

# -----------------------------------------------------------------------------
# Unit Tests: Text Chunking
# -----------------------------------------------------------------------------
def test_chunking_exact_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # length 36
    # chunk_size=15, overlap=5 -> step=10
    # chunk 0: [0:15]  -> "ABCDEFGHIJKLMNO"
    # chunk 1: [10:25] -> "KLMNOPQRSTUVWXY"
    # chunk 2: [20:35] -> "UVWXYZ012345678"
    # chunk 3: [30:36] -> "456789"
    chunks = chunk_text(text, chunk_size=15, chunk_overlap=5)
    assert len(chunks) == 4
    assert chunks[0] == "ABCDEFGHIJKLMNO"
    assert chunks[1] == "KLMNOPQRSTUVWXY"
    assert chunks[2] == "UVWXYZ012345678"
    assert chunks[3] == "456789"

def test_chunking_short_text():
    text = "Short financial statement."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunking_empty_text():
    assert chunk_text("", chunk_size=100, chunk_overlap=20) == []
    assert chunk_text("   \n\t  ", chunk_size=100, chunk_overlap=20) == []

def test_chunking_invalid_arguments():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        chunk_text("test", chunk_size=50, chunk_overlap=50)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        chunk_text("test", chunk_size=50, chunk_overlap=-5)

# -----------------------------------------------------------------------------
# Unit Tests: Reciprocal Rank Fusion (Hand-Computed Verification)
# -----------------------------------------------------------------------------
def test_reciprocal_rank_fusion_hand_computed():
    """
    Verifies RRF mathematically against a hand-computed 3-document test case:
    List 1 (Dense): doc_A (rank 1), doc_B (rank 2)
    List 2 (Sparse): doc_B (rank 1), doc_C (rank 2)
    
    With k=60:
      doc_A = 1/(60+1)                = 1/61            ≈ 0.01639344
      doc_B = 1/(60+2) + 1/(60+1)     = 1/62 + 1/61     ≈ 0.03252248
      doc_C = 1/(60+2)                = 1/62            ≈ 0.01612903
      
    Expected order: doc_B (1st), doc_A (2nd), doc_C (3rd).
    """
    list_dense = [
        {"chunk_id": "doc_A", "text": "Text A"},
        {"chunk_id": "doc_B", "text": "Text B"}
    ]
    list_sparse = [
        {"chunk_id": "doc_B", "text": "Text B"},
        {"chunk_id": "doc_C", "text": "Text C"}
    ]

    fused = reciprocal_rank_fusion([list_dense, list_sparse], k=60)

    assert len(fused) == 3
    assert fused[0]["chunk_id"] == "doc_B"
    assert fused[1]["chunk_id"] == "doc_A"
    assert fused[2]["chunk_id"] == "doc_C"

    expected_score_b = (1.0 / 62) + (1.0 / 61)
    expected_score_a = 1.0 / 61
    expected_score_c = 1.0 / 62

    assert abs(fused[0]["rrf_score"] - expected_score_b) < 1e-7
    assert abs(fused[1]["rrf_score"] - expected_score_a) < 1e-7
    assert abs(fused[2]["rrf_score"] - expected_score_c) < 1e-7

