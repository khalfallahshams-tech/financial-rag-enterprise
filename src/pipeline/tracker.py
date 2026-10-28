"""
MLflow Experiment Tracking and Run Management.
Logs parameters, metrics, confidence intervals, cost, latency, Git commit, and artifacts.
Also imports historical legacy runs into MLflow for visual comparison in MLflow UI.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_NAME = "financial_rag_enterprise"


def get_git_commit_hash() -> str:
    """Retrieve current Git commit SHA, or 'unknown' if not in git."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for MLflow compliance (removes '@', '%', etc.)."""
    import re
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\. /]", "_", name)
    return re.sub(r"_+", "_", cleaned).strip("_")


class MLflowTracker:
    """
    Manages structured MLflow tracking for the Financial RAG pipeline.
    """
    def __init__(self, experiment_name: str = EXPERIMENT_NAME):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def log_retrieval_run(
        self,
        config_name: str,
        metrics_dict: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        artifact_path: Optional[Path] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Log a retrieval benchmark configuration as a distinct MLflow run."""
        with mlflow.start_run(run_name=f"Retrieval_{config_name}"):
            # Tags
            run_tags = {
                "component": "retrieval",
                "git_commit": get_git_commit_hash(),
            }
            if tags:
                run_tags.update(tags)
            mlflow.set_tags(run_tags)

            # Parameters
            run_params = params or {}
            run_params["config_name"] = config_name
            mlflow.log_params(run_params)

            # Metrics
            flat_metrics = {}
            for k, v in metrics_dict.items():
                sanitized_k = sanitize_metric_name(k)
                if isinstance(v, (int, float)):
                    flat_metrics[sanitized_k] = float(v)
                elif isinstance(v, dict) and "mean" in v:
                    flat_metrics[f"{sanitized_k}_mean"] = float(v["mean"])
                    if "ci_95" in v:
                        flat_metrics[f"{sanitized_k}_ci_lower"] = float(v["ci_95"][0])
                        flat_metrics[f"{sanitized_k}_ci_upper"] = float(v["ci_95"][1])
            mlflow.log_metrics(flat_metrics)

            # Artifacts
            if artifact_path and Path(artifact_path).exists():
                mlflow.log_artifact(str(artifact_path))

    def log_generation_run(
        self,
        generation_results: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        artifact_path: Optional[Path] = None,
    ):
        """Log generation and guardrail evaluation run."""
        with mlflow.start_run(run_name="Generation_Defensive_Gemini"):
            run_tags = {
                "component": "generation_and_guardrails",
                "git_commit": get_git_commit_hash(),
            }
            mlflow.set_tags(run_tags)

            run_params = params or {}
            run_params["sample_size"] = generation_results.get("sample_size", 0)
            mlflow.log_params(run_params)

            # Log metrics
            metrics = {}
            for metric_name, m_data in generation_results.get("metrics", {}).items():
                s_name = sanitize_metric_name(metric_name)
                metrics[f"{s_name}_mean"] = m_data["mean"]
                metrics[f"{s_name}_ci_lower"] = m_data["ci_95"][0]
                metrics[f"{s_name}_ci_upper"] = m_data["ci_95"][1]

            if "token_usage" in generation_results:
                tu = generation_results["token_usage"]
                metrics["estimated_cost_usd"] = tu.get("estimated_cost_usd", 0.0)
                metrics["prompt_tokens"] = tu.get("estimated_prompt_tokens", 0)
                metrics["completion_tokens"] = tu.get("estimated_completion_tokens", 0)

            if "latency_ms" in generation_results:
                lat = generation_results["latency_ms"]
                metrics["latency_mean_ms"] = lat.get("mean", 0.0)
                metrics["latency_p95_ms"] = lat.get("p95", 0.0)

            mlflow.log_metrics(metrics)

            if artifact_path and Path(artifact_path).exists():
                mlflow.log_artifact(str(artifact_path))

    def import_legacy_experiments(self) -> int:
        """
        Imports historical benchmark files from results/ as tagged MLflow runs.
        Returns the number of imported runs.
        """
        results_dir = BASE_DIR / "results"
        imported_count = 0

        # 1. BEIR Retrieval Benchmark
        beir_file = results_dir / "beir_retrieval_benchmark_results.json"
        if beir_file.exists():
            with open(beir_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cfg_name, metrics in data.items():
                self.log_retrieval_run(
                    config_name=cfg_name,
                    metrics_dict=metrics,
                    params={"suite": "Suite 1 Academic (30 queries)"},
                    artifact_path=beir_file,
                    tags={"legacy": "true", "phase": "1"},
                )
                imported_count += 1

        # 2. Candidate Depth Sweep
        sweep_file = results_dir / "candidate_depth_sweep.json"
        if sweep_file.exists():
            with open(sweep_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k_name, row in data.items():
                with mlflow.start_run(run_name=f"Ablation_CandidateDepth_{k_name}"):
                    mlflow.set_tags({"component": "ablation", "legacy": "true", "phase": "1"})
                    mlflow.log_params({"candidate_depth_k": row["Candidate_K_Per_Channel"]})
                    mlflow.log_metrics({
                        "Hit_1": row["Hit@1"],
                        "Hit_3": row["Hit@3"],
                        "MRR": row["MRR"],
                        "Avg_Latency_ms": row["Avg_Latency_ms"],
                        "P95_Latency_ms": row["P95_Latency_ms"],
                    })
                    mlflow.log_artifact(str(sweep_file))
                    imported_count += 1

        # 3. Acronym Expansion Benchmark
        acronym_file = results_dir / "acronym_expansion_benchmark.json"
        if acronym_file.exists():
            with open(acronym_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            with mlflow.start_run(run_name="Ablation_Acronym_Expansion"):
                mlflow.set_tags({"component": "query_expansion", "legacy": "true", "phase": "1"})
                mlflow.log_metrics({
                    "mean_baseline_confidence": data["mean_baseline_confidence"],
                    "mean_expanded_confidence": data["mean_expanded_confidence"],
                    "net_confidence_lift": data["net_confidence_lift"],
                })
                mlflow.log_artifact(str(acronym_file))
                imported_count += 1

        # 4. Generation Evaluation
        gen_file = results_dir / "generation_evaluation_results.json"
        if gen_file.exists():
            with open(gen_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.log_generation_run(
                generation_results=data,
                params={"model": "gemini-3.5-flash-lite", "phase": "1"},
                artifact_path=gen_file,
            )
            imported_count += 1

        return imported_count

