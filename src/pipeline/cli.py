"""
Pipeline Command Line Interface (CLI).
Supports:
  python -m src.pipeline.cli all [--force]
  python -m src.pipeline.cli eval [--force]
  python -m src.pipeline.cli stage <stage_name> [--force]
  python -m src.pipeline.cli import-legacy
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.stages import (
    stage_verify_data,
    stage_chunk,
    stage_embed,
    stage_build_index,
    stage_evaluate_retrieval,
    stage_evaluate_generation,
    stage_build_report,
)
from src.pipeline.tracker import MLflowTracker

STAGES_MAP = {
    "verify_data": stage_verify_data,
    "chunk": stage_chunk,
    "embed": stage_embed,
    "build_index": stage_build_index,
    "evaluate_retrieval": stage_evaluate_retrieval,
    "evaluate_generation": stage_evaluate_generation,
    "build_report": stage_build_report,
}


def run_full_pipeline(force: bool = False, gen_limit: int = 50):
    """Executes stages 1 through 7 end-to-end."""
    print("=" * 80)
    print("RUNNING END-TO-END FINANCIAL RAG PIPELINE (make pipeline)")
    print("=" * 80)
    t0 = time.time()

    stage_verify_data(force=force)
    stage_chunk(force=force)
    stage_embed(force=force)
    stage_build_index(force=force)
    stage_evaluate_retrieval(force=force)
    stage_evaluate_generation(force=force, limit=gen_limit)
    stage_build_report(force=force)

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE in {total_time:.1f} seconds! All stages executed or cached.")
    print("=" * 80)


def run_eval_pipeline(force: bool = False, gen_limit: int = 50):
    """Executes evaluation stages (retrieval, generation, build_report)."""
    print("=" * 80)
    print("RUNNING EVALUATION PIPELINE (make eval)")
    print("=" * 80)
    t0 = time.time()

    stage_evaluate_retrieval(force=force)
    stage_evaluate_generation(force=force, limit=gen_limit)
    stage_build_report(force=force)

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"EVALUATION PIPELINE COMPLETE in {total_time:.1f} seconds!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Financial RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline execution command")

    # Command: all / pipeline
    p_all = subparsers.add_parser("all", help="Run end-to-end pipeline (stages 1-7)")
    p_all.add_argument("--force", action="store_true", help="Bypass cache and force rerun all stages")
    p_all.add_argument("--gen-limit", type=int, default=50, help="Max generation evaluation questions")

    # Command: eval
    p_eval = subparsers.add_parser("eval", help="Rerun evaluation stages only (stages 5-7)")
    p_eval.add_argument("--force", action="store_true", help="Bypass cache and force rerun evaluation")
    p_eval.add_argument("--gen-limit", type=int, default=50, help="Max generation evaluation questions")

    # Command: stage
    p_stage = subparsers.add_parser("stage", help="Run a single pipeline stage")
    p_stage.add_argument("stage_name", choices=list(STAGES_MAP.keys()), help="Name of the stage to execute")
    p_stage.add_argument("--force", action="store_true", help="Bypass cache for this stage")

    # Command: import-legacy
    subparsers.add_parser("import-legacy", help="Import legacy benchmark results into MLflow")

    args = parser.parse_args()

    if args.command in ("all", "pipeline", None):
        force = getattr(args, "force", False)
        limit = getattr(args, "gen_limit", 50)
        run_full_pipeline(force=force, gen_limit=limit)
    elif args.command == "eval":
        run_eval_pipeline(force=args.force, gen_limit=args.gen_limit)
    elif args.command == "stage":
        stage_fn = STAGES_MAP[args.stage_name]
        stage_fn(force=args.force)
    elif args.command == "import-legacy":
        tracker = MLflowTracker()
        n = tracker.import_legacy_experiments()
        print(f"Successfully imported {n} historical benchmark runs into MLflow.")


if __name__ == "__main__":
    main()

