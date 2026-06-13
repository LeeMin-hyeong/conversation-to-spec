from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model_runner import MLXModelRunner
from app.utils import normalize_text

CATEGORIES = (
    ("functional_requirements", "functional requirement"),
    ("non_functional_requirements", "non-functional requirement"),
    ("constraints", "constraint"),
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


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with",
        "be", "is", "are", "it", "this", "that", "should", "shall", "must", "can",
        "could", "will", "would", "system", "website", "app", "application",
    }
    return {t for t in normalize_text(text).split() if len(t) > 2 and t not in stop}


def _similarity(left: str, right: str) -> float:
    lt = _tokens(left)
    rt = _tokens(right)
    if not lt or not rt:
        return 0.0
    return 2 * len(lt & rt) / (len(lt) + len(rt))


def _source_similarity(left: list[str], right: list[str]) -> float:
    ls = {x for x in left if x}
    rs = {x for x in right if x}
    if not ls or not rs:
        return 0.0
    return len(ls & rs) / max(len(ls), len(rs))


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("text", "")).strip()
    return str(getattr(item, "text", "")).strip()


def _item_sources(item: Any) -> list[str]:
    if isinstance(item, dict):
        return [str(x).strip() for x in item.get("source_units", []) if str(x).strip()]
    return []


def _load_pred(pred_dir: Path, sample_id: str) -> dict[str, Any]:
    return json.loads((pred_dir / f"{sample_id}_pred.json").read_text(encoding="utf-8"))


def _source_excerpt(sample: dict[str, Any], source_ids: list[str]) -> str:
    units = sample.get("conversation_units") or []
    by_id = {str(u.get("id", "")): str(u.get("text", "")) for u in units if isinstance(u, dict)}
    selected = [f"{sid}: {by_id[sid]}" for sid in source_ids if sid in by_id]
    if selected:
        return "\n".join(selected)
    return str(sample.get("conversation_text", ""))


def _candidate_pairs(
    sample: dict[str, Any],
    pred_spec: dict[str, Any],
    *,
    run_label: str,
    top_k: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    sid = str(sample["id"])
    for category, category_label in CATEGORIES:
        gold_items = [g for g in sample.get("gold", {}).get(category, []) if _item_text(g)]
        pred_items = [p for p in pred_spec.get(category, []) if _item_text(p)]
        if not gold_items or not pred_items:
            continue
        for pred_index, pred in enumerate(pred_items):
            scored = []
            pred_text = _item_text(pred)
            pred_sources = _item_sources(pred)
            for gold_index, gold in enumerate(gold_items):
                gold_text = _item_text(gold)
                gold_sources = _item_sources(gold)
                score = 0.75 * _similarity(pred_text, gold_text) + 0.25 * _source_similarity(pred_sources, gold_sources)
                scored.append((score, gold_index, gold))
            scored.sort(key=lambda x: x[0], reverse=True)
            for rank, (candidate_score, gold_index, gold) in enumerate(scored[:top_k], start=1):
                gold_sources = _item_sources(gold)
                source_ids = sorted(set(pred_sources + gold_sources))
                pairs.append(
                    {
                        "id": f"{run_label}:{sid}:{category}:p{pred_index}:g{gold_index}",
                        "run": run_label,
                        "sample_id": sid,
                        "category": category,
                        "category_label": category_label,
                        "pred_index": pred_index,
                        "gold_index": gold_index,
                        "candidate_rank": rank,
                        "candidate_score": round(candidate_score, 4),
                        "pred_text": pred_text,
                        "gold_text": _item_text(gold),
                        "pred_source_units": pred_sources,
                        "gold_source_units": gold_sources,
                        "source_excerpt": _source_excerpt(sample, source_ids),
                    }
                )
    return pairs


def _build_prompt(pair: dict[str, Any]) -> str:
    return f"""You are an impartial evaluator for software requirements extraction.

Evaluate whether the predicted requirement captures the same underlying requirement as the gold requirement. Use the source conversation only as grounding context.

Category: {pair['category_label']}

Source conversation excerpt:
{pair['source_excerpt']}

Gold requirement:
{pair['gold_text']}

Predicted requirement:
{pair['pred_text']}

Rubric:
- MATCH: The predicted requirement expresses the same actionable requirement as the gold requirement. Minor wording differences are acceptable.
- PARTIAL: The predicted requirement captures part of the gold requirement but misses an important condition, actor, scope, or quality target.
- NO_MATCH: The predicted requirement is a different requirement, unsupported by the source, or too vague to count as the gold requirement.

Return JSON only with this schema:
{{"verdict":"MATCH|PARTIAL|NO_MATCH","reason":"short reason","unsupported_detail":true|false,"type_error":true|false}}
"""


def _parse_judgment(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    payload: dict[str, Any] = {}
    if match:
        try:
            payload = json.loads(match.group(0))
        except Exception:
            payload = {}
    verdict = str(payload.get("verdict", "")).upper().strip()
    if verdict not in {"MATCH", "PARTIAL", "NO_MATCH"}:
        upper = cleaned.upper()
        if "NO_MATCH" in upper or "NO MATCH" in upper:
            verdict = "NO_MATCH"
        elif "PARTIAL" in upper:
            verdict = "PARTIAL"
        elif "MATCH" in upper:
            verdict = "MATCH"
        else:
            verdict = "NO_MATCH"
    return {
        "verdict": verdict,
        "reason": str(payload.get("reason", cleaned[:240])).strip()[:500],
        "unsupported_detail": bool(payload.get("unsupported_detail", False)),
        "type_error": bool(payload.get("type_error", False)),
        "raw_output": cleaned,
    }


def _verdict_score(verdict: str) -> float:
    if verdict == "MATCH":
        return 1.0
    if verdict == "PARTIAL":
        return 0.5
    return 0.0


def _aggregate(samples: list[dict[str, Any]], judgments: list[dict[str, Any]], run_label: str, pred_dir: Path) -> dict[str, Any]:
    by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in judgments:
        if row["run"] == run_label:
            by_key.setdefault((row["sample_id"], row["category"], row["pred_index"]), []).append(row)

    category_stats: dict[str, dict[str, float]] = {}
    total_pred = total_gold = 0
    total_credit = 0.0
    unmatched_pred = 0
    partial_pred = 0
    matched_gold_keys: set[tuple[str, str, int]] = set()

    for category, _ in CATEGORIES:
        pred_count = gold_count = 0
        credit = 0.0
        cat_unmatched = 0
        cat_partial = 0
        cat_matched_gold: set[tuple[str, str, int]] = set()
        for sample in samples:
            sid = str(sample["id"])
            pred_spec = _load_pred(pred_dir, sid)
            gold_items = [g for g in sample.get("gold", {}).get(category, []) if _item_text(g)]
            pred_items = [p for p in pred_spec.get(category, []) if _item_text(p)]
            gold_count += len(gold_items)
            pred_count += len(pred_items)
            for pred_index, _pred in enumerate(pred_items):
                candidates = by_key.get((sid, category, pred_index), [])
                candidates.sort(key=lambda r: (_verdict_score(r["verdict"]), r.get("candidate_score", 0.0)), reverse=True)
                best = candidates[0] if candidates else None
                score = _verdict_score(best["verdict"]) if best else 0.0
                if best and score > 0:
                    gold_key = (sid, category, int(best["gold_index"]))
                    if gold_key in cat_matched_gold:
                        score = 0.0
                    else:
                        cat_matched_gold.add(gold_key)
                        matched_gold_keys.add(gold_key)
                if score == 0:
                    cat_unmatched += 1
                elif score == 0.5:
                    cat_partial += 1
                credit += score
        precision = credit / pred_count if pred_count else 0.0
        recall = credit / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        category_stats[category] = {
            "predictions": float(pred_count),
            "gold": float(gold_count),
            "match_credit": credit,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "unmatched_prediction_rate": cat_unmatched / pred_count if pred_count else 0.0,
            "partial_prediction_rate": cat_partial / pred_count if pred_count else 0.0,
        }
        total_pred += pred_count
        total_gold += gold_count
        total_credit += credit
        unmatched_pred += cat_unmatched
        partial_pred += cat_partial

    macro_f1 = sum(v["f1"] for v in category_stats.values()) / len(CATEGORIES)
    precision = total_credit / total_pred if total_pred else 0.0
    recall = total_credit / total_gold if total_gold else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "run": run_label,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall,
        "predictions": total_pred,
        "gold": total_gold,
        "match_credit": total_credit,
        "unmatched_prediction_rate": unmatched_pred / total_pred if total_pred else 0.0,
        "partial_prediction_rate": partial_pred / total_pred if total_pred else 0.0,
        "by_category": category_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/prometheus_judge"))
    parser.add_argument("--model", default="mlx-community/prometheus-7b-v2.0-8bit")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit_samples:
        samples = samples[: args.limit_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[dict[str, Any]] = []
    for run_label, pred_dir in DEFAULT_RUNS.items():
        for sample in samples:
            pairs.extend(_candidate_pairs(sample, _load_pred(pred_dir, str(sample["id"])), run_label=run_label, top_k=args.top_k))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    pairs_path = args.output_dir / "judge_pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")

    judgments_path = args.output_dir / "judge_results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if args.resume and judgments_path.exists():
        for line in judgments_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row

    runner = MLXModelRunner(args.model)
    runner.prepare()
    started = time.perf_counter()
    with judgments_path.open("a", encoding="utf-8") as f:
        for index, pair in enumerate(pairs, start=1):
            if pair["id"] in done:
                continue
            prompt = _build_prompt(pair)
            raw = runner.generate(
                prompt,
                {
                    "max_new_tokens": 160,
                    "temperature": 0.0,
                    "do_sample": False,
                    "seed": 7,
                    "stop_sequences": ["\n\n\n"],
                },
            )
            parsed = _parse_judgment(raw)
            row = {**pair, **parsed, "latency_sec": runner.last_generation_info.get("latency_sec")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{index}/{len(pairs)}] {pair['id']} -> {row['verdict']}", flush=True)

    judgments = []
    for line in judgments_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            judgments.append(json.loads(line))
    summaries = {
        run_label: _aggregate(samples, judgments, run_label, pred_dir)
        for run_label, pred_dir in DEFAULT_RUNS.items()
    }
    summary = {
        "judge_model": args.model,
        "dataset_size": len(samples),
        "top_k": args.top_k,
        "pair_count": len(pairs),
        "completed_judgments": len(judgments),
        "elapsed_sec": time.perf_counter() - started,
        "runs": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
