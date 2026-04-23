from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from app.evaluation import build_comparison_table, evaluate_model, load_eval_dataset
from app.model_runner import HFModelRunner, MockModelRunner
from app.pipeline import ConversationToSpecPipeline
from app.progress import ConsoleProgressReporter
from app.prompt_builder import load_prompt_config
from app.utils import ensure_dir, load_yaml_file, slugify, write_json_file, write_text_file


MODELS_CONFIG_PATH = Path("configs/models.yaml")
PROMPTS_CONFIG_PATH = Path("configs/prompts.yaml")


def _resolve_model_alias(input_name: str, models_config: dict[str, Any]) -> tuple[str, str]:
    models = models_config.get("models", {})
    if input_name in models:
        return input_name, str(models[input_name]["hf_repo_id"])

    for alias, cfg in models.items():
        if str(cfg.get("hf_repo_id", "")) == input_name:
            return alias, input_name

    # Direct repository id support.
    return slugify(input_name), input_name


def _build_pipeline(
    *,
    use_mock: bool,
    model_name: str | None,
    prompt_config: dict[str, Any],
    generation_config: dict[str, Any],
    models_config: dict[str, Any],
) -> tuple[str, ConversationToSpecPipeline]:
    if use_mock:
        runner = MockModelRunner()
        return "mock", ConversationToSpecPipeline(
            runner=runner,
            prompt_config=prompt_config,
            generation_config=generation_config,
        )

    if not model_name:
        raise ValueError("A model name is required unless --mock is used.")

    alias, hf_repo_id = _resolve_model_alias(model_name, models_config)
    runner = HFModelRunner(hf_repo_id)
    return alias, ConversationToSpecPipeline(
        runner=runner,
        prompt_config=prompt_config,
        generation_config=generation_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversation-to-Spec CLI")
    parser.add_argument("--input", type=str, help="Path to input transcript (.txt/.md)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--model", type=str, help="Model alias or Hugging Face repo id")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock model")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--dataset", type=str, help="Evaluation dataset path (JSON)")
    parser.add_argument("--all-models", action="store_true", help="Evaluate all configured models")
    return parser.parse_args()


def _run_single(args: argparse.Namespace, models_config: dict[str, Any], prompt_config: dict[str, Any]) -> int:
    if not args.input:
        print("Error: --input is required for single-run mode.")
        return 2
    if args.mock and args.model:
        print("Error: use either --mock or --model, not both.")
        return 2
    if not args.mock and not args.model:
        args.model = str(models_config.get("default_model", ""))

    generation_config = models_config.get("generation", {})
    output_dir = ensure_dir(Path(args.output))
    reporter = ConsoleProgressReporter()

    try:
        _, pipeline = _build_pipeline(
            use_mock=args.mock,
            model_name=args.model,
            prompt_config=prompt_config,
            generation_config=generation_config,
            models_config=models_config,
        )
        run = pipeline.run_file(Path(args.input), output_dir, progress_reporter=reporter)
        json_path = run.output_json_path or (output_dir / "spec.json")
        md_path = run.output_md_path or (output_dir / "spec.md")
        print(f"Saved JSON: {json_path}")
        print(f"Saved Markdown: {md_path}")
        print(f"Status: {run.status} (retry_count={run.retry_count})")
        if run.semantic_warnings:
            print(f"Semantic warnings: {len(run.semantic_warnings)}")
        if not run.success:
            error_log = output_dir / "error.log"
            write_text_file(error_log, run.error_message or "Invalid structured output.")
            print(
                "Error: Failed to generate a valid spec output. "
                f"Details recorded at {error_log}: {run.error_message or 'invalid output'}"
            )
            return 1
        return 0
    except Exception as exc:
        error_log = output_dir / "error.log"
        write_text_file(error_log, str(exc))
        print(
            "Error: Failed to generate a valid spec output. "
            f"Details recorded at {error_log}: {exc}"
        )
        return 1


def _run_evaluate(
    args: argparse.Namespace, models_config: dict[str, Any], prompt_config: dict[str, Any]
) -> int:
    if not args.dataset:
        print("Error: --dataset is required when --evaluate is set.")
        return 2
    if args.mock and args.all_models:
        print("Error: --mock cannot be combined with --all-models.")
        return 2
    if not args.mock and not args.all_models and not args.model:
        args.model = str(models_config.get("default_model", ""))

    dataset = load_eval_dataset(Path(args.dataset))
    generation_config = models_config.get("generation", {})
    eval_root = ensure_dir(Path("eval_output"))
    reporter = ConsoleProgressReporter()

    if args.all_models:
        model_aliases = list(models_config.get("compare_models", []))
        if not model_aliases:
            model_aliases = list(models_config.get("models", {}).keys())
        if not model_aliases:
            print("Error: no models configured in configs/models.yaml.")
            return 2

        all_reports: dict[str, dict[str, Any]] = {}
        for model_index, alias in enumerate(model_aliases, start=1):
            model_output_dir = ensure_dir(eval_root / slugify(alias))
            try:
                reporter.message(f"Model {model_index}/{len(model_aliases)} [{alias}] started")
                resolved_alias, pipeline = _build_pipeline(
                    use_mock=False,
                    model_name=alias,
                    prompt_config=prompt_config,
                    generation_config=generation_config,
                    models_config=models_config,
                )
                report = evaluate_model(
                    model_label=resolved_alias,
                    pipeline=pipeline,
                    samples=dataset,
                    output_dir=model_output_dir,
                    progress_reporter=reporter,
                )
                reporter.message(f"Model {model_index}/{len(model_aliases)} [{alias}] finished")
            except Exception as exc:
                report = {
                    "model": alias,
                    "metrics": {
                        "sample_count": len(dataset),
                        "functional_f1": 0.0,
                        "non_functional_f1": 0.0,
                        "requirement_type_macro_f1": 0.0,
                        "open_question_recall": 0.0,
                        "follow_up_question_coverage": 0.0,
                        "hallucination_rate": 0.0,
                        "schema_validity_rate": 0.0,
                        "avg_latency_sec": 0.0,
                    },
                    "error": str(exc),
                }
                write_text_file(model_output_dir / "error.log", str(exc))
            all_reports[alias] = report

        comparison_json = eval_root / "comparison_results.json"
        comparison_md = eval_root / "comparison_table.md"
        write_json_file(comparison_json, all_reports)
        write_text_file(comparison_md, build_comparison_table(all_reports))
        print(f"Saved comparison JSON: {comparison_json}")
        print(f"Saved comparison table: {comparison_md}")
        return 0

    use_mock = bool(args.mock)
    model_name = "mock" if use_mock else args.model
    output_dir = ensure_dir(eval_root / slugify(str(model_name)))
    try:
        resolved_alias, pipeline = _build_pipeline(
            use_mock=use_mock,
            model_name=model_name,
            prompt_config=prompt_config,
            generation_config=generation_config,
            models_config=models_config,
        )
        report = evaluate_model(
            model_label=resolved_alias,
            pipeline=pipeline,
            samples=dataset,
            output_dir=output_dir,
            progress_reporter=reporter,
        )
        print(f"Saved metrics: {output_dir / 'metrics.json'}")
        print(f"Model: {report.get('model')}")
        return 0
    except Exception as exc:
        write_text_file(output_dir / "error.log", str(exc))
        print(f"Error: evaluation failed for {model_name}: {exc}")
        return 1


def main() -> int:
    args = parse_args()
    models_config = load_yaml_file(MODELS_CONFIG_PATH)
    prompt_config = load_prompt_config(PROMPTS_CONFIG_PATH)

    if args.evaluate:
        return _run_evaluate(args, models_config, prompt_config)
    return _run_single(args, models_config, prompt_config)


if __name__ == "__main__":
    sys.exit(main())
