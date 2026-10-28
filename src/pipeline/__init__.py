"""
Unified Pipeline and Orchestration Package for Financial RAG.
Exports:
- FinancialRAGPipeline: Production serving orchestrator (Two-Stage Hybrid + Guardrails)
- Pipeline stages: verify_data, chunk, embed, build_index, evaluate_retrieval, evaluate_generation, build_report
- CLI runners: run_full_pipeline, run_eval_pipeline
"""

from src.pipeline.orchestrator import FinancialRAGPipeline
from src.pipeline.stages import (
    stage_verify_data,
    stage_chunk,
    stage_embed,
    stage_build_index,
    stage_evaluate_retrieval,
    stage_evaluate_generation,
    stage_build_report,
)
from src.pipeline.cli import run_full_pipeline, run_eval_pipeline

__all__ = [
    "FinancialRAGPipeline",
    "stage_verify_data",
    "stage_chunk",
    "stage_embed",
    "stage_build_index",
    "stage_evaluate_retrieval",
    "stage_evaluate_generation",
    "stage_build_report",
    "run_full_pipeline",
    "run_eval_pipeline",
]

