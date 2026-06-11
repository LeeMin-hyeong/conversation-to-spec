from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.evaluation import build_comparison_table, evaluate_model, load_eval_dataset
from app.model_runner import HFModelRunner
from app.pipeline import ConversationToSpecPipeline
from app.progress import ConsoleProgressReporter
from app.prompt_builder import load_prompt_config
from app.utils import ensure_dir, load_yaml_file, slugify, write_json_file, write_text_file


MODELS_CONFIG_PATH = Path("configs/models.yaml")
PROMPTS_CONFIG_PATH = Path("configs/prompts.yaml")
PROMPT_STYLES = ("zero_shot", "few_shot")
VERIFY_MODES = ("off", "heuristic", "llm", "minicheck")
PIPELINE_MODE = "single_shot"


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
    model_name: str,
    prompt_config: dict[str, Any],
    generation_config: dict[str, Any],
    models_config: dict[str, Any],
    prompt_style: str = "few_shot",
    verify_mode: str = "minicheck",
    repair_on_fail: bool = False,
) -> tuple[str, ConversationToSpecPipeline]:
    if not model_name:
        raise ValueError("A model name is required.")

    alias, hf_repo_id = _resolve_model_alias(model_name, models_config)
    runner = HFModelRunner(hf_repo_id)
    return alias, ConversationToSpecPipeline(
        runner=runner,
        prompt_config=prompt_config,
        generation_config=generation_config,
        pipeline_mode=PIPELINE_MODE,
        prompt_style=prompt_style,
        verify_mode=verify_mode,
        repair_on_fail=repair_on_fail,
    )


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_id(args: argparse.Namespace) -> str:
    return str(args.run_id or "").strip() or _timestamp_run_id()


def _experiment_run_root(args: argparse.Namespace) -> Path:
    return ensure_dir(Path(args.experiment_root) / _run_id(args))


def _model_repo_id(model_name: str | None, models_config: dict[str, Any]) -> str | None:
    if not model_name:
        return None
    _, repo_id = _resolve_model_alias(model_name, models_config)
    return repo_id


def _run_metadata(
    *,
    args: argparse.Namespace,
    model_alias: str,
    model_name: str,
    models_config: dict[str, Any],
    generation_config: dict[str, Any],
    dataset_path: Path | None,
    output_dir: Path,
    experiment_run_root: Path | None,
) -> dict[str, Any]:
    prompt_path = PROMPTS_CONFIG_PATH
    model_path = MODELS_CONFIG_PATH
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pipeline_mode": PIPELINE_MODE,
        "prompt_style": args.prompt_style,
        "verify_mode": args.verify_mode,
        "repair_on_fail": bool(args.repair_on_fail),
        "model_alias": model_alias,
        "model_input": model_name,
        "hf_repo_id": _model_repo_id(model_name, models_config),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_sha256": _sha256_file(dataset_path) if dataset_path else None,
        "prompt_config_path": str(prompt_path),
        "prompt_config_sha256": _sha256_file(prompt_path),
        "models_config_path": str(model_path),
        "models_config_sha256": _sha256_file(model_path),
        "generation_config": generation_config,
        "output_dir": str(output_dir),
        "experiment_run_root": str(experiment_run_root) if experiment_run_root else None,
        "argv": sys.argv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversation-to-Spec CLI")
    parser.add_argument("--input", type=str, help="Path to input transcript (.txt/.md)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--model", type=str, help="Model alias or Hugging Face repo id")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--dataset", type=str, help="Evaluation dataset path (JSON)")
    parser.add_argument("--all-models", action="store_true", help="Evaluate configured comparison models")
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default="few_shot",
        help="Prompt style for the single-call specification generator.",
    )
    parser.add_argument(
        "--verify-mode",
        choices=VERIFY_MODES,
        default="minicheck",
        help="Verification mode for generated requirements.",
    )
    parser.add_argument(
        "--repair-on-fail",
        action="store_true",
        help="Repair only unsupported or incomplete requirements after verification.",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Write evaluation artifacts under experiments/runs/<timestamp>/.",
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        default="experiments/runs",
        help="Root directory for timestamped experiment runs.",
    )
    parser.add_argument("--run-id", type=str, help="Optional timestamp/run id to reuse.")
    return parser.parse_args()


def _default_model(models_config: dict[str, Any]) -> str:
    model_name = str(models_config.get("default_model", "")).strip()
    if not model_name:
        raise ValueError("configs/models.yaml must define default_model.")
    return model_name


def _run_single(args: argparse.Namespace, models_config: dict[str, Any], prompt_config: dict[str, Any]) -> int:
    if not args.input:
        print("Error: --input is required for single-run mode.")
        return 2
    if not args.model:
        args.model = _default_model(models_config)

    generation_config = models_config.get("generation", {})
    model_alias, _ = _resolve_model_alias(args.model, models_config)
    run_id = _run_id(args)
    output_dir = ensure_dir(
        Path(args.output) / slugify(f"{run_id}__{model_alias}__{PIPELINE_MODE}")
    )
    reporter = ConsoleProgressReporter()

    try:
        resolved_alias, pipeline = _build_pipeline(
            model_name=args.model,
            prompt_config=prompt_config,
            generation_config=generation_config,
            models_config=models_config,
            prompt_style=args.prompt_style,
            verify_mode=args.verify_mode,
            repair_on_fail=bool(args.repair_on_fail),
        )
        print(f"Run output directory: {output_dir}")
        run = pipeline.run_file(Path(args.input), output_dir, progress_reporter=reporter)
        json_path = run.output_json_path or (output_dir / "spec.json")
        md_path = run.output_md_path or (output_dir / "spec.md")
        verification_json_path = run.verification_report_json_path or (
            output_dir / "verification_report.json"
        )
        verification_md_path = run.verification_report_md_path or (
            output_dir / "verification_report.md"
        )
        print(f"Saved JSON: {json_path}")
        print(f"Saved Markdown: {md_path}")
        print(f"Saved verification JSON: {verification_json_path}")
        print(f"Saved verification Markdown: {verification_md_path}")
        print(f"Status: {run.status} (llm_calls={run.num_llm_calls})")
        if run.semantic_warnings:
            print(f"Verification warnings: {len(run.semantic_warnings)}")
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
    args: argparse.Namespace,
    models_config: dict[str, Any],
    prompt_config: dict[str, Any],
) -> int:
    if not args.dataset:
        print("Error: --dataset is required when --evaluate is set.")
        return 2
    if not args.all_models and not args.model:
        args.model = _default_model(models_config)

    dataset_path = Path(args.dataset)
    dataset = load_eval_dataset(dataset_path)
    generation_config = models_config.get("generation", {})
    experiment_run_root = _experiment_run_root(args) if args.experiment else None
    run_id = _run_id(args)
    eval_root = ensure_dir(
        (experiment_run_root / "eval_output")
        if experiment_run_root
        else Path("eval_output") / run_id
    )
    reporter = ConsoleProgressReporter()

    model_names: list[str]
    if args.all_models:
        model_names = [str(alias) for alias in models_config.get("compare_models", [])]
        if not model_names:
            model_names = list(models_config.get("models", {}).keys())
        if not model_names:
            print("Error: no models configured in configs/models.yaml.")
            return 2
    else:
        model_names = [str(args.model)]

    all_reports: dict[str, dict[str, Any]] = {}
    for model_index, model_name in enumerate(model_names, start=1):
        output_dir = ensure_dir(eval_root / slugify(f"{model_name}__{PIPELINE_MODE}"))
        try:
            reporter.message(f"Model {model_index}/{len(model_names)} [{model_name}] started")
            resolved_alias, pipeline = _build_pipeline(
                model_name=model_name,
                prompt_config=prompt_config,
                generation_config=generation_config,
                models_config=models_config,
                prompt_style=args.prompt_style,
                verify_mode=args.verify_mode,
                repair_on_fail=bool(args.repair_on_fail),
            )
            metadata = _run_metadata(
                args=args,
                model_alias=resolved_alias,
                model_name=model_name,
                models_config=models_config,
                generation_config=generation_config,
                dataset_path=dataset_path,
                output_dir=output_dir,
                experiment_run_root=experiment_run_root,
            )
            report = evaluate_model(
                model_label=resolved_alias,
                pipeline=pipeline,
                samples=dataset,
                output_dir=output_dir,
                progress_reporter=reporter,
                run_metadata=metadata,
            )
            reporter.message(f"Model {model_index}/{len(model_names)} [{model_name}] finished")
        except Exception as exc:
            report = {
                "model": model_name,
                "metrics": {
                    "sample_count": len(dataset),
                    "functional_f1": 0.0,
                    "non_functional_f1": 0.0,
                    "requirement_type_macro_f1": 0.0,
                    "schema_validity_rate": 0.0,
                    "avg_latency_sec": 0.0,
                    "num_llm_calls": 0.0,
                },
                "error": str(exc),
            }
            write_text_file(output_dir / "error.log", str(exc))
        all_reports[str(report.get("model", model_name))] = report

    if len(all_reports) == 1:
        only_report = next(iter(all_reports.values()))
        print(f"Saved metrics: {output_dir / 'metrics.json'}")
        if experiment_run_root:
            print(f"Saved experiment run: {experiment_run_root}")
        print(f"Model: {only_report.get('model')}")
        return 0

    comparison_json = eval_root / "comparison_results.json"
    comparison_md = eval_root / "comparison_table.md"
    write_json_file(comparison_json, all_reports)
    write_text_file(comparison_md, build_comparison_table(all_reports))
    print(f"Saved comparison JSON: {comparison_json}")
    print(f"Saved comparison table: {comparison_md}")
    return 0


def main() -> int:
    args = parse_args()
    models_config = load_yaml_file(MODELS_CONFIG_PATH)
    prompt_config = load_prompt_config(PROMPTS_CONFIG_PATH)

    if args.evaluate:
        return _run_evaluate(args, models_config, prompt_config)
    return _run_single(args, models_config, prompt_config)


if __name__ == "__main__":
    sys.exit(main())
