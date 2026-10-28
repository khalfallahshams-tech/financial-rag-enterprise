from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest

from src.generation.llm import DefensiveGenerator
from src.pipeline.tracker import MLflowTracker, sanitize_metric_name
from src.pipeline.stages import stage_chunk, stage_embed, stage_build_index
from src.pipeline.cli import build_parser

def test_sanitize_metric_name():
    assert sanitize_metric_name("nDCG@10") == "nDCG_10"
    assert sanitize_metric_name("Recall@100_mean") == "Recall_100_mean"
    assert sanitize_metric_name("clean_name") == "clean_name"

def test_mlflow_tracker_initialization():
    tracker = MLflowTracker(experiment_name="test_experiment")
    assert tracker.experiment_name == "test_experiment"

def test_mlflow_tracker_log_run():
    tracker = MLflowTracker(experiment_name="test_experiment")
    with patch("mlflow.start_run") as mock_start:
        mock_run = MagicMock()
        mock_start.return_value.__enter__.return_value = mock_run

        tracker.log_retrieval_run(
            config_name="Hybrid_Rerank",
            metrics_dict={"ndcg_10": {"mean": 0.490, "ci_low": 0.348, "ci_high": 0.636}},
            params={"dense_k": 8, "bm25_k": 8},
            tags={"branch": "portfolio-build"}
        )
        assert mock_start.called

def test_generator_fallback_without_key():
    """Verifies that DefensiveGenerator raises ValueError when API key is empty."""
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        DefensiveGenerator(api_key="")

def test_pipeline_stages_inspection():
    """Exercises idempotent stage functions to verify cached state behavior."""
    r_chunk = stage_chunk()
    assert r_chunk["status"] == "VALID"
    assert r_chunk["total_indexed_chunks"] > 0

    r_embed = stage_embed()
    assert r_embed["status"] in ("VALID", "READY")

    r_index = stage_build_index()
    assert r_index["status"] == "VALID"

def test_mlflow_import_legacy_experiments():
    """Exercises legacy experiments import into MLflow with mocked logging calls."""
    tracker = MLflowTracker(experiment_name="test_experiment")
    with patch.object(tracker, "log_retrieval_run") as mock_ret, \
         patch.object(tracker, "log_generation_run") as mock_gen, \
         patch("mlflow.start_run"), \
         patch("mlflow.set_tags"), \
         patch("mlflow.log_params"), \
         patch("mlflow.log_metrics"), \
         patch("mlflow.log_artifact"):
        count = tracker.import_legacy_experiments()
        assert count >= 5
        assert mock_ret.called
        assert mock_gen.called

def test_mlflow_tracker_log_generation_run():
    tracker = MLflowTracker(experiment_name="test_experiment")
    with patch("mlflow.start_run"), patch("mlflow.set_tags"), patch("mlflow.log_params"), patch("mlflow.log_metrics"), patch("mlflow.log_artifact"):
        mock_gen_results = {
            "faithfulness": {"mean": 0.98, "ci_low": 0.96, "ci_high": 1.0},
            "answer_relevancy": {"mean": 0.96, "ci_low": 0.93, "ci_high": 0.99},
            "gate_accuracy": {"mean": 0.94, "ci_low": 0.86, "ci_high": 1.0},
            "cost_summary": {"cost_per_query_usd": 0.000028, "total_cost_usd": 0.0014}
        }
        tracker.log_generation_run(mock_gen_results, params={"model": "gemini-3.5-flash-lite"})

def test_cli_parser_dispatch():
    parser = build_parser()
    args_all = parser.parse_args(["all"])
    assert args_all.command == "all"

    args_eval = parser.parse_args(["eval"])
    assert args_eval.command == "eval"

    args_stage = parser.parse_args(["stage", "verify_data"])
    assert args_stage.command == "stage"
    assert args_stage.stage_name == "verify_data"

def test_pipeline_cli_runners():
    from src.pipeline.cli import run_full_pipeline, run_eval_pipeline
    with patch("src.pipeline.cli.stage_verify_data"), \
         patch("src.pipeline.cli.stage_chunk"), \
         patch("src.pipeline.cli.stage_embed"), \
         patch("src.pipeline.cli.stage_build_index"), \
         patch("src.pipeline.cli.stage_evaluate_retrieval"), \
         patch("src.pipeline.cli.stage_evaluate_generation"), \
         patch("src.pipeline.cli.stage_build_report"):
        run_full_pipeline(force=True, gen_limit=2)
        run_eval_pipeline(force=True, gen_limit=2)


