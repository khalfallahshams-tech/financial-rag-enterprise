import time
from typing import Any

from google import genai

from src.config import settings

DEFENSIVE_PROMPT_TEMPLATE = """You are a strict, factual financial intelligence assistant.
Your instructions:
1. Answer the user question using ONLY the facts explicitly stated in the provided SOURCES below.
2. For EVERY factual claim or sentence, cite the source using the exact format [Source X: doc_id] at the end of the claim.
3. If the provided sources do NOT contain sufficient information to answer the question, or if the question asks about a non-existent regulation, form, or rule, you MUST explicitly state: "The provided context does not contain sufficient information to answer this question."
4. NEVER speculate, assume, or extrapolate beyond what is directly stated.

SOURCES:
{context_block}

USER QUESTION: {question}

RESPONSE:"""

class DefensiveGenerator:
    """
    Stage-4 Defensive Generator.
    Interacts with Gemini 3.5 Flash Lite using our proven Defensive Prompt Contract.
    Features automatic exponential-backoff retries for production resilience against transient API spikes.
    """
    def __init__(self, api_key: str = settings.gemini_api_key, model_name: str = settings.gemini_model_name):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or settings.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, question: str, sources: list[dict[str, Any]]) -> tuple[str, float]:
        """
        Generates a defensive, citation-attributed financial response.
        Returns:
            (answer_text, latency_ms)
        """
        # Format sources block
        context_blocks = []
        for s in sources:
            tag = s.get("source_tag", "Source")
            cid = s.get("chunk_id", "doc_unknown")
            text = s.get("text", "")
            context_blocks.append(f"[{tag}: {cid}]\n{text}")

        formatted_context = "\n\n".join(context_blocks)
        prompt = DEFENSIVE_PROMPT_TEMPLATE.format(
            context_block=formatted_context,
            question=question
        )

        t0 = time.time()
        last_exception = None

        # Exponential backoff retry loop
        for attempt in range(settings.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                latency_ms = (time.time() - t0) * 1000
                return response.text.strip(), latency_ms
            except Exception as e:
                last_exception = e
                if attempt < settings.max_retries - 1:
                    sleep_time = settings.retry_delay_seconds * (2 ** attempt)
                    time.sleep(sleep_time)

        # Fallback if all retries exhausted
        latency_ms = (time.time() - t0) * 1000
        return f"Service temporarily unavailable: {last_exception}", latency_ms

