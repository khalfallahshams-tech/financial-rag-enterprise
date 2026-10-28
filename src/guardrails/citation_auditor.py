import re
from typing import List, Dict, Any
from src.schemas import CitationAuditReport

class CitationAuditor:
    """
    Automated Citation Attribution Engine.
    Audits generated answer text against the retrieved chunks provided in the prompt.
    Verifies that 100% of bracketed source tags point to authentic database chunk IDs.
    """
    @staticmethod
    def audit(text: str, retrieved_sources: List[Dict[str, Any]]) -> CitationAuditReport:
        valid_source_tags = {s.get("source_tag", "") for s in retrieved_sources if s.get("source_tag")}
        valid_chunk_ids = {s.get("chunk_id", "") for s in retrieved_sources if s.get("chunk_id")}

        # Extract bracketed citations like [Source 1: doc_6301_c0] or [Source 1]
        raw_citations = re.findall(
            r'\[(?:Source\s*\d+(?::\s*[^\]]+)?|doc_\d+_c\d+)\]',
            text,
            flags=re.IGNORECASE
        )

        valid_citations = 0
        hallucinated_citations = 0

        for cite in raw_citations:
            cleaned = cite.strip('[]')
            matched = any(tag.lower() in cleaned.lower() for tag in valid_source_tags)
            if not matched:
                matched = any(cid.lower() in cleaned.lower() for cid in valid_chunk_ids)

            if matched:
                valid_citations += 1
            else:
                hallucinated_citations += 1

        total_citations = len(raw_citations)
        precision = (valid_citations / total_citations) if total_citations > 0 else 1.0

        # Sentence-level coverage analysis
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]
        cited_sentences = sum(
            1 for s in sentences if re.search(r'\[(?:Source|doc_)', s, flags=re.IGNORECASE)
        )
        coverage = (cited_sentences / len(sentences)) if sentences else 1.0

        return CitationAuditReport(
            total_citations=total_citations,
            valid_citations=valid_citations,
            hallucinated_citations=hallucinated_citations,
            citation_precision=precision,
            total_sentences=len(sentences),
            cited_sentences=cited_sentences,
            sentence_coverage=coverage
        )

