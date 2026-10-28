from typing import List

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping sliding window chunks.
    Ensures deterministic, lossless chunking with boundary safety.
    
    Args:
        text: Input raw text string.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.
        
    Returns:
        List of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size.")

    clean_text = text.strip()
    if not clean_text:
        return []

    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = clean_text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start += step

    return chunks

