import sys
import argparse
from src.pipeline import FinancialRAGPipeline
from src.schemas import QueryRequest

def run_cli():
    parser = argparse.ArgumentParser(description="Enterprise Financial RAG Pipeline CLI")
    parser.add_argument("--query", "-q", type=str, help="Financial query to evaluate")
    parser.add_argument("--no-cache", action="store_true", help="Bypass memory cache")
    args = parser.parse_args()

    print("Initializing Enterprise Financial RAG Pipeline...")
    pipeline = FinancialRAGPipeline()
    print("Pipeline ready!\n")

    def process_query(question: str, use_cache: bool):
        req = QueryRequest(question=question, use_cache=use_cache)
        resp = pipeline.run(req)

        print("=" * 75)
        print(f"QUERY: {resp.question}")
        print("=" * 75)
        print(f"GATE DECISION:    {resp.gate_decision.value}")
        print(f"CONFIDENCE SCORE: {resp.confidence_score:.4f}")
        print(f"CACHE STATUS:     {'[CACHE HIT]' if resp.is_cached else '[FRESH RUN]'}")
        print(f"LATENCY:          {resp.total_latency_ms:.1f} ms (Retrieval: {resp.retrieval_latency_ms:.1f}ms | Rerank: {resp.rerank_latency_ms:.1f}ms | LLM: {resp.llm_latency_ms:.1f}ms)")
        print("-" * 75)
        print(f"ANSWER:\n{resp.answer}\n")

        if resp.citations:
            print("RETRIEVED CITATIONS:")
            for c in resp.citations:
                print(f"  * [{c.source_tag}: {c.chunk_id}] (Score: {c.score:.2f}) -> {c.snippet[:100]}...")
            print(f"\nCITATION AUDIT: Precision: {resp.citation_audit.citation_precision*100:.1f}% | Valid: {resp.citation_audit.valid_citations}/{resp.citation_audit.total_citations} | Sentence Coverage: {resp.citation_audit.sentence_coverage*100:.1f}%")
        print("=" * 75 + "\n")

    if args.query:
        process_query(args.query, use_cache=not args.no_cache)
    else:
        print("Entering Interactive Mode (type 'exit' or 'quit' to stop):")
        while True:
            try:
                user_q = input("\nEnter Financial Question > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "q"]:
                    print("Exiting RAG CLI. Goodbye!")
                    break
                process_query(user_q, use_cache=not args.no_cache)
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    run_cli()

