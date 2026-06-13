from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CATEGORIES = (
    ("functional_requirements", "FR"),
    ("non_functional_requirements", "NFR"),
    ("constraints", "CON"),
)

DEFAULT_DATASET = Path("experiments/version_compare/all_eval_datasets.json")
DEFAULT_RUNS = {
    "v0.1.0": Path(
        "experiments/version_compare/v010_all_mlx_qwen3_4b_instruct/"
        "eval_output/qwen3_4b_mlx_4bit__chain/predictions"
    ),
    "v0.1.1": Path(
        "experiments/version_compare/v011_all_mlx_qwen3_4b_instruct/"
        "eval_output/qwen3_4b_mlx_4bit__single_shot/predictions"
    ),
}


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("text", "")).strip()
    return str(getattr(item, "text", "")).strip()


def _load_pred(pred_dir: Path, sample_id: str) -> dict[str, Any]:
    return json.loads((pred_dir / f"{sample_id}_pred.json").read_text(encoding="utf-8"))


def _build_pairs(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for run_label, pred_dir in DEFAULT_RUNS.items():
        for sample in samples:
            sample_id = str(sample["id"])
            pred_spec = _load_pred(pred_dir, sample_id)
            gold = sample.get("gold", {})
            for category, short_category in CATEGORIES:
                gold_items = [_item_text(item) for item in gold.get(category, []) if _item_text(item)]
                pred_items = [_item_text(item) for item in pred_spec.get(category, []) if _item_text(item)]
                units.append(
                    {
                        "run": run_label,
                        "sample_id": sample_id,
                        "category": category,
                        "category_short": short_category,
                        "gold_count": len(gold_items),
                        "pred_count": len(pred_items),
                    }
                )
                for pred_index, pred_text in enumerate(pred_items):
                    for gold_index, gold_text in enumerate(gold_items):
                        rows.append(
                            {
                                "run": run_label,
                                "sample_id": sample_id,
                                "category": category,
                                "category_short": short_category,
                                "pred_index": pred_index,
                                "gold_index": gold_index,
                                "pred_text": pred_text,
                                "gold_text": gold_text,
                            }
                        )
    return rows, units


def _select_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _score_pairs(rows: list[dict[str, Any]], model_name: str, batch_size: int) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _select_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    scored: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["pred_text"] for row in batch],
                [row["gold_text"] for row in batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            if not isinstance(scores, list):
                scores = [float(scores)]
            for row, score in zip(batch, scores):
                scored.append({**row, "cross_encoder_score": float(score)})
            print(f"Scored {min(start + batch_size, len(rows))}/{len(rows)} pairs", flush=True)
    return scored


def _greedy_gold_coverage(rows: list[dict[str, Any]], gold_count: int) -> dict[str, float | int]:
    if gold_count <= 0:
        return {"coverage_score": 0.0, "matched_count": 0, "score_sum": 0.0}
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    score_sum = 0.0
    for row in sorted(rows, key=lambda item: float(item["cross_encoder_score"]), reverse=True):
        gold_index = int(row["gold_index"])
        pred_index = int(row["pred_index"])
        if gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        score_sum += float(row["cross_encoder_score"])
    return {
        "coverage_score": score_sum / gold_count,
        "matched_count": len(used_gold),
        "score_sum": score_sum,
    }


def _aggregate(scored_rows: list[dict[str, Any]], units: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_run: dict[str, list[float]] = defaultdict(list)
    rows_by_unit: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        score = float(row["cross_encoder_score"])
        run = row["run"]
        category = row["category"]
        grouped[(run, category)].append(score)
        by_run[run].append(score)
        rows_by_unit[(run, row["sample_id"], category)].append(row)

    coverage_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    coverage_by_run: dict[str, list[float]] = defaultdict(list)
    gold_counts: dict[tuple[str, str], int] = defaultdict(int)
    gold_counts_by_run: dict[str, int] = defaultdict(int)
    pred_counts: dict[tuple[str, str], int] = defaultdict(int)
    pred_counts_by_run: dict[str, int] = defaultdict(int)
    matched_counts: dict[tuple[str, str], int] = defaultdict(int)
    matched_counts_by_run: dict[str, int] = defaultdict(int)
    for unit in units:
        gold_count = int(unit["gold_count"])
        pred_count = int(unit["pred_count"])
        if gold_count <= 0:
            continue
        run = unit["run"]
        category = unit["category"]
        unit_rows = rows_by_unit.get((run, unit["sample_id"], category), [])
        coverage_result = _greedy_gold_coverage(unit_rows, gold_count)
        coverage = float(coverage_result["coverage_score"])
        matched_count = int(coverage_result["matched_count"])
        coverage_scores[(run, category)].append(coverage)
        coverage_by_run[run].append(coverage)
        gold_counts[(run, category)] += gold_count
        gold_counts_by_run[run] += gold_count
        pred_counts[(run, category)] += pred_count
        pred_counts_by_run[run] += pred_count
        matched_counts[(run, category)] += matched_count
        matched_counts_by_run[run] += matched_count

    runs: dict[str, Any] = {}
    for run in sorted(set(by_run) | set(coverage_by_run)):
        scores = by_run.get(run, [])
        category_rows: dict[str, Any] = {}
        category_means: list[float] = []
        category_coverages: list[float] = []
        for category, short_category in CATEGORIES:
            category_scores = grouped.get((run, category), [])
            avg = mean(category_scores) if category_scores else 0.0
            category_coverage_scores = coverage_scores.get((run, category), [])
            coverage_avg = mean(category_coverage_scores) if category_coverage_scores else 0.0
            category_means.append(avg)
            category_coverages.append(coverage_avg)
            category_rows[category] = {
                "label": short_category,
                "pair_count": len(category_scores),
                "gold_count": gold_counts.get((run, category), 0),
                "pred_count": pred_counts.get((run, category), 0),
                "matched_count": matched_counts.get((run, category), 0),
                "missing_gold_count": gold_counts.get((run, category), 0)
                - matched_counts.get((run, category), 0),
                "mean_score": avg,
                "gold_normalized_coverage_score": coverage_avg,
            }
        runs[run] = {
            "all_pair_count": len(scores),
            "gold_count": gold_counts_by_run.get(run, 0),
            "pred_count": pred_counts_by_run.get(run, 0),
            "matched_count": matched_counts_by_run.get(run, 0),
            "missing_gold_count": gold_counts_by_run.get(run, 0)
            - matched_counts_by_run.get(run, 0),
            "all_pair_mean_score": mean(scores) if scores else 0.0,
            "category_macro_mean_score": mean(category_means) if category_means else 0.0,
            "gold_normalized_coverage_score": mean(coverage_by_run[run]) if coverage_by_run.get(run) else 0.0,
            "gold_normalized_category_macro_score": (
                mean(category_coverages) if category_coverages else 0.0
            ),
            "by_category": category_rows,
        }
    return {"runs": runs}


def _write_outputs(
    output_dir: Path,
    model_name: str,
    scored_rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model_name,
        "score_definition": (
            "Mean sigmoid-normalized cross-encoder score over all same-category "
            "predicted requirement and gold requirement pairs."
        ),
        "gold_normalized_coverage_definition": (
            "For each sample and category, greedily match predicted requirements to gold "
            "requirements by cross-encoder score. Divide the matched score sum by the "
            "number of gold requirements, so missing predictions receive zero credit."
        ),
        **_aggregate(scored_rows, units),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "pair_scores.jsonl").open("w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "category",
                "pair_count",
                "gold_count",
                "pred_count",
                "matched_count",
                "missing_gold_count",
                "all_pair_mean_score",
                "gold_normalized_coverage_score",
            ]
        )
        for run, payload in summary["runs"].items():
            writer.writerow(
                [
                    run,
                    "ALL",
                    payload["all_pair_count"],
                    payload["gold_count"],
                    payload["pred_count"],
                    payload["matched_count"],
                    payload["missing_gold_count"],
                    payload["all_pair_mean_score"],
                    payload["gold_normalized_coverage_score"],
                ]
            )
            writer.writerow(
                [
                    run,
                    "CATEGORY_MACRO",
                    "",
                    "",
                    "",
                    "",
                    "",
                    payload["category_macro_mean_score"],
                    payload["gold_normalized_category_macro_score"],
                ]
            )
            for category, category_payload in payload["by_category"].items():
                writer.writerow(
                    [
                        run,
                        category_payload["label"],
                        category_payload["pair_count"],
                        category_payload["gold_count"],
                        category_payload["pred_count"],
                        category_payload["matched_count"],
                        category_payload["missing_gold_count"],
                        category_payload["mean_score"],
                        category_payload["gold_normalized_coverage_score"],
                    ]
                )

    lines = [
        "# Cross-Encoder Semantic Scores",
        "",
        "## Gold-Normalized Coverage",
        "",
        "| Run | Matched/Gold | Missing gold | Gold-normalized coverage | Category macro coverage | FR | NFR | CON |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run, payload in summary["runs"].items():
        by_category = payload["by_category"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run,
                    f"{payload['matched_count']}/{payload['gold_count']}",
                    str(payload["missing_gold_count"]),
                    f"{payload['gold_normalized_coverage_score']:.3f}",
                    f"{payload['gold_normalized_category_macro_score']:.3f}",
                    f"{by_category['functional_requirements']['gold_normalized_coverage_score']:.3f}",
                    f"{by_category['non_functional_requirements']['gold_normalized_coverage_score']:.3f}",
                    f"{by_category['constraints']['gold_normalized_coverage_score']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## All-Pair Mean",
            "",
            "| Run | ALL pair mean | Category macro mean | FR | NFR | CON |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run, payload in summary["runs"].items():
        by_category = payload["by_category"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run,
                    f"{payload['all_pair_mean_score']:.3f}",
                    f"{payload['category_macro_mean_score']:.3f}",
                    f"{by_category['functional_requirements']['mean_score']:.3f}",
                    f"{by_category['non_functional_requirements']['mean_score']:.3f}",
                    f"{by_category['constraints']['mean_score']:.3f}",
                ]
            )
            + " |"
        )
    (output_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default="cross-encoder/stsb-roberta-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cross_encoder_scores"))
    parser.add_argument("--limit-samples", type=int, default=0)
    args = parser.parse_args()

    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit_samples:
        samples = samples[: args.limit_samples]
    rows, units = _build_pairs(samples)
    print(f"Built {len(rows)} same-category gold-pred pairs from {len(samples)} samples.")
    scored_rows = _score_pairs(rows, args.model, args.batch_size)
    _write_outputs(args.output_dir, args.model, scored_rows, units)
    print((args.output_dir / "summary_table.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
