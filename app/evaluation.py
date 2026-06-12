from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

import json
import re

from app.pipeline import ConversationToSpecPipeline
from app.progress import NullProgressReporter
from app.schemas import SpecOutput
from app.utils import ensure_dir, model_dump_compat, normalize_text, write_json_file, write_text_file
from app.verifier import format_verification_report_markdown


SEMANTIC_MATCH_THRESHOLD = 0.42
SEMANTIC_SOURCE_OVERLAP_THRESHOLD = 0.5
SEMANTIC_MIN_TEXT_WITH_SOURCE_MATCH = 0.12
SEMANTIC_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "be",
    "is",
    "are",
    "it",
    "this",
    "that",
    "should",
    "shall",
    "must",
    "can",
    "could",
    "will",
    "would",
    "from",
    "by",
    "as",
    "at",
    "system",
    "website",
    "site",
    "app",
    "application",
}

SEMANTIC_TOKEN_ALIASES = {
    "booking": {"book", "reserve", "reservation"},
    "book": {"booking", "reserve", "reservation"},
    "reserve": {"book", "booking", "reservation"},
    "reservation": {"book", "booking", "reserve"},
    "quick": {"quickly", "fast", "performance"},
    "quickly": {"quick", "fast", "performance"},
    "fast": {"quick", "quickly", "performance"},
    "phone": {"phones", "mobile"},
    "phones": {"phone", "mobile"},
    "mobile": {"phone", "phones"},
    "secure": {"security"},
    "security": {"secure"},
    "reliable": {"reliability"},
    "reliability": {"reliable"},
    "simple": {"easy", "usable", "usability"},
    "easy": {"simple", "usable", "usability"},
    "ios": {"iphone"},
    "iphone": {"ios"},
    "workshop": {"workshops"},
    "workshops": {"workshop"},
}


@dataclass
class SampleStatus:
    sample_id: str
    success: bool
    json_parse_ok: bool
    pydantic_validation_ok: bool
    latency_sec: float
    final_status: str = ""
    retry_count: int = 0
    retry_success: bool = False
    retry_recovery: bool = False
    fallback_rescue: bool = False
    semantic_warning: bool = False
    stage_failure: Optional[str] = None
    stage_retry_counts: dict[str, int] = None
    stage_stats: dict[str, Any] = None
    error: Optional[str] = None


def load_eval_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON list.")
    return payload


def _extract_texts(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    output: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            continue
        if text:
            output.append(text)
    return output


def _extract_eval_records(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    records: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            source_units: list[str] = []
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            source_units = [
                str(source_id).strip()
                for source_id in item.get("source_units", [])
                if str(source_id).strip()
            ]
        else:
            text = str(getattr(item, "text", "")).strip()
            source_units = [
                str(source_id).strip()
                for source_id in getattr(item, "source_units", [])
                if str(source_id).strip()
            ]
        if text:
            records.append({"text": text, "source_units": source_units})
    return records


def _prf(pred_items: set[tuple[str, str]], gold_items: set[tuple[str, str]]) -> dict[str, float]:
    tp = len(pred_items & gold_items)
    fp = len(pred_items - gold_items)
    fn = len(gold_items - pred_items)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if not gold_items else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _semantic_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in normalize_text(text).split():
        if not token or token in SEMANTIC_STOPWORDS or len(token) <= 2:
            continue
        tokens.add(token)
        if token.endswith("ing") and len(token) > 5:
            tokens.add(token[:-3])
        if token.endswith("s") and len(token) > 4:
            tokens.add(token[:-1])
        tokens.update(SEMANTIC_TOKEN_ALIASES.get(token, set()))
    return tokens


def _semantic_similarity(left: str, right: str) -> float:
    left_tokens = _semantic_tokens(left)
    right_tokens = _semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return (2 * overlap) / (len(left_tokens) + len(right_tokens))


def _source_similarity(left_sources: list[str], right_sources: list[str]) -> float:
    left = {source_id for source_id in left_sources if source_id}
    right = {source_id for source_id in right_sources if source_id}
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def _semantic_prf(
    samples: list[dict[str, Any]],
    predicted_specs: dict[str, SpecOutput],
    category_key: str,
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    for sample in samples:
        sid = str(sample["id"])
        gold_records = _extract_eval_records(sample.get("gold", {}).get(category_key, []))
        spec = predicted_specs.get(sid)
        pred_records = (
            _extract_eval_records(list(getattr(spec, category_key, [])))
            if spec is not None
            else []
        )

        matched_gold: set[int] = set()
        for pred in pred_records:
            best_index = -1
            best_score = 0.0
            for index, gold in enumerate(gold_records):
                if index in matched_gold:
                    continue
                text_score = _semantic_similarity(pred["text"], gold["text"])
                source_score = _source_similarity(pred["source_units"], gold["source_units"])
                score = text_score
                if (
                    source_score >= SEMANTIC_SOURCE_OVERLAP_THRESHOLD
                    and text_score >= SEMANTIC_MIN_TEXT_WITH_SOURCE_MATCH
                ):
                    score = max(score, SEMANTIC_MATCH_THRESHOLD)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index >= 0 and best_score >= SEMANTIC_MATCH_THRESHOLD:
                tp += 1
                matched_gold.add(best_index)
            else:
                fp += 1
        fn += max(0, len(gold_records) - len(matched_gold))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if fn == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _gold_category_set(
    samples: Iterable[dict[str, Any]], category_key: str, normalized: bool = False
) -> set[tuple[str, str]]:
    values: set[tuple[str, str]] = set()
    for sample in samples:
        sid = str(sample["id"])
        gold = sample.get("gold", {})
        for text in _extract_texts(gold.get(category_key, [])):
            key = normalize_text(text) if normalized else text
            values.add((sid, key))
    return values


def _pred_category_set(
    predicted_specs: dict[str, SpecOutput], category_key: str, normalized: bool = False
) -> set[tuple[str, str]]:
    values: set[tuple[str, str]] = set()
    for sid, spec in predicted_specs.items():
        items = getattr(spec, category_key)
        for item in items:
            text = item.text.strip()
            if not text:
                continue
            key = normalize_text(text) if normalized else text
            values.add((sid, key))
    return values


def _pred_requirement_records(predicted_specs: dict[str, SpecOutput]) -> list[tuple[SpecOutput, Any]]:
    records: list[tuple[SpecOutput, Any]] = []
    for spec in predicted_specs.values():
        for item in (
            list(spec.functional_requirements)
            + list(spec.non_functional_requirements)
            + list(spec.constraints)
        ):
            records.append((spec, item))
    return records


def _non_empty_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _traceability_ok(spec: SpecOutput, item: Any) -> bool:
    valid_unit_ids = {unit.id for unit in spec.conversation_units}
    source_units = [str(source_id).strip() for source_id in getattr(item, "source_units", []) if str(source_id).strip()]
    evidence_spans = _non_empty_strings(getattr(item, "evidence_spans", []))
    return bool(source_units and evidence_spans and all(source_id in valid_unit_ids for source_id in source_units))


def _quality_gate_ok(item: Any) -> bool:
    checks = getattr(item, "quality_checks", None)
    return bool(
        checks
        and checks.is_atomic
        and checks.is_testable
        and checks.has_clear_actor
        and checks.has_traceable_evidence
    )


def _requirement_quality_metrics(predicted_specs: dict[str, SpecOutput]) -> dict[str, float]:
    records = _pred_requirement_records(predicted_specs)
    total = len(records)
    if not total:
        return {
            "requirement_count": 0.0,
            "acceptance_criteria_coverage": 0.0,
            "evidence_span_coverage": 0.0,
            "traceability_coverage": 0.0,
            "quality_gate_pass_rate": 0.0,
            "high_ambiguity_rate": 0.0,
            "groundedness_rate": 0.0,
            "unsupported_requirement_rate": 0.0,
            "verification_pass_rate": 0.0,
        }

    with_acceptance = sum(
        1 for _, item in records if _non_empty_strings(getattr(item, "acceptance_criteria", []))
    )
    with_evidence = sum(
        1 for _, item in records if _non_empty_strings(getattr(item, "evidence_spans", []))
    )
    traceable = sum(1 for spec, item in records if _traceability_ok(spec, item))
    gate_pass = sum(1 for _, item in records if _quality_gate_ok(item))
    high_ambiguity = sum(
        1
        for _, item in records
        if str(getattr(getattr(item, "quality_checks", None), "ambiguity_risk", "")).strip().lower()
        == "high"
    )
    grounded = sum(
        1
        for _, item in records
        if getattr(item, "verification", None)
        and item.verification.verdict in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
    )
    unsupported = sum(
        1
        for _, item in records
        if getattr(item, "verification", None)
        and item.verification.verdict in {"UNSUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
    )
    verification_pass = sum(
        1
        for _, item in records
        if getattr(item, "verification", None)
        and item.verification.verdict == "SUPPORTED"
    )
    return {
        "requirement_count": float(total),
        "acceptance_criteria_coverage": with_acceptance / total,
        "evidence_span_coverage": with_evidence / total,
        "traceability_coverage": traceable / total,
        "quality_gate_pass_rate": gate_pass / total,
        "high_ambiguity_rate": high_ambiguity / total,
        "groundedness_rate": grounded / total,
        "unsupported_requirement_rate": unsupported / total,
        "verification_pass_rate": verification_pass / total,
    }


def compute_metrics(
    samples: list[dict[str, Any]],
    predicted_specs: dict[str, SpecOutput],
    sample_statuses: dict[str, SampleStatus],
) -> dict[str, Any]:
    fr_gold = _gold_category_set(samples, "functional_requirements", normalized=False)
    nfr_gold = _gold_category_set(samples, "non_functional_requirements", normalized=False)
    con_gold = _gold_category_set(samples, "constraints", normalized=False)
    oq_gold = _gold_category_set(samples, "open_questions", normalized=False)
    fu_gold = _gold_category_set(samples, "follow_up_questions", normalized=False)
    note_gold = _gold_category_set(samples, "notes", normalized=False)

    fr_pred = _pred_category_set(predicted_specs, "functional_requirements", normalized=False)
    nfr_pred = _pred_category_set(predicted_specs, "non_functional_requirements", normalized=False)
    con_pred = _pred_category_set(predicted_specs, "constraints", normalized=False)
    oq_pred = _pred_category_set(predicted_specs, "open_questions", normalized=False)
    fu_pred = _pred_category_set(predicted_specs, "follow_up_questions", normalized=False)
    note_pred = _pred_category_set(predicted_specs, "notes", normalized=False)

    fr_metrics = _prf(fr_pred, fr_gold)
    nfr_metrics = _prf(nfr_pred, nfr_gold)
    con_metrics = _prf(con_pred, con_gold)
    oq_metrics = _prf(oq_pred, oq_gold)
    fu_metrics = _prf(fu_pred, fu_gold)
    note_metrics = _prf(note_pred, note_gold)
    fr_semantic = _semantic_prf(samples, predicted_specs, "functional_requirements")
    nfr_semantic = _semantic_prf(samples, predicted_specs, "non_functional_requirements")
    con_semantic = _semantic_prf(samples, predicted_specs, "constraints")
    open_question_semantic = _semantic_prf(samples, predicted_specs, "open_questions")
    follow_up_semantic = _semantic_prf(samples, predicted_specs, "follow_up_questions")

    category_f1 = [
        fr_metrics["f1"],
        nfr_metrics["f1"],
        con_metrics["f1"],
        oq_metrics["f1"],
        fu_metrics["f1"],
        note_metrics["f1"],
    ]
    macro_f1 = sum(category_f1) / len(category_f1)
    semantic_requirement_macro_f1 = (
        fr_semantic["f1"] + nfr_semantic["f1"] + con_semantic["f1"]
    ) / 3

    all_req_pred = fr_pred | nfr_pred | con_pred
    all_req_gold = fr_gold | nfr_gold | con_gold
    exact_unsupported = all_req_pred - all_req_gold
    exact_hallucination_rate = (
        len(exact_unsupported) / len(all_req_pred) if all_req_pred else 0.0
    )
    semantic_fp = fr_semantic["fp"] + nfr_semantic["fp"] + con_semantic["fp"]
    semantic_pred_total = semantic_fp + fr_semantic["tp"] + nfr_semantic["tp"] + con_semantic["tp"]
    hallucination_rate = semantic_fp / semantic_pred_total if semantic_pred_total else 0.0
    requirement_quality_metrics = _requirement_quality_metrics(predicted_specs)

    sample_count = len(sample_statuses)
    if sample_count:
        parse_rate = (
            sum(1 for status in sample_statuses.values() if status.json_parse_ok) / sample_count
        )
        validation_rate = (
            sum(1 for status in sample_statuses.values() if status.pydantic_validation_ok)
            / sample_count
        )
        full_schema_rate = (
            sum(
                1
                for status in sample_statuses.values()
                if status.json_parse_ok and status.pydantic_validation_ok
            )
            / sample_count
        )
        usable_output_rate = (
            sum(1 for status in sample_statuses.values() if status.success) / sample_count
        )
        retry_success_rate = (
            sum(1 for status in sample_statuses.values() if status.retry_success) / sample_count
        )
        retry_recovery_rate = (
            sum(1 for status in sample_statuses.values() if getattr(status, "retry_recovery", False))
            / sample_count
        )
        fallback_rescue_rate = (
            sum(1 for status in sample_statuses.values() if getattr(status, "fallback_rescue", False))
            / sample_count
        )
        semantic_warning_rate = (
            sum(1 for status in sample_statuses.values() if status.semantic_warning) / sample_count
        )
    else:
        parse_rate = 0.0
        validation_rate = 0.0
        full_schema_rate = 0.0
        usable_output_rate = 0.0
        retry_success_rate = 0.0
        retry_recovery_rate = 0.0
        fallback_rescue_rate = 0.0
        semantic_warning_rate = 0.0

    stage_failure_counts: dict[str, int] = {}
    stage_retry_total = 0
    stage_retry_observed = 0
    stage4_open_question_values: list[float] = []
    stage5_followup_values: list[float] = []
    num_llm_calls_values: list[float] = []
    repair_trigger_total = 0.0
    repair_success_total = 0.0
    repair_denominator = 0.0
    for status in sample_statuses.values():
        if status.stage_failure:
            stage_failure_counts[status.stage_failure] = (
                stage_failure_counts.get(status.stage_failure, 0) + 1
            )
        retry_map = status.stage_retry_counts or {}
        if retry_map:
            stage_retry_total += sum(int(v) for v in retry_map.values())
            stage_retry_observed += 1
        stats_map = status.stage_stats or {}
        if "stage_4_open_question_count" in stats_map:
            try:
                stage4_open_question_values.append(float(stats_map["stage_4_open_question_count"]))
            except Exception:
                pass
        if "stage_5_follow_up_count" in stats_map:
            try:
                stage5_followup_values.append(float(stats_map["stage_5_follow_up_count"]))
            except Exception:
                pass
        if "num_llm_calls" in stats_map:
            try:
                num_llm_calls_values.append(float(stats_map["num_llm_calls"]))
            except Exception:
                pass
        try:
            repair_trigger_total += float(stats_map.get("repair_trigger_count", 0.0))
            repair_success_total += float(stats_map.get("repair_success_count", 0.0))
            repair_denominator += float(stats_map.get("requirement_quality_enrichment_count", 0.0))
        except Exception:
            pass

    observed_latencies = [s.latency_sec for s in sample_statuses.values()]
    avg_latency = mean(observed_latencies) if observed_latencies else 0.0
    constraint_warning_count = 0
    for spec in predicted_specs.values():
        for warning in list(getattr(spec, "verification_warnings", []) or []):
            warning_text = str(warning).lower()
            if warning_text.startswith("constraints:") or warning_text.startswith("raw_constraints:"):
                constraint_warning_count += 1

    fr_norm = _prf(
        _pred_category_set(predicted_specs, "functional_requirements", normalized=True),
        _gold_category_set(samples, "functional_requirements", normalized=True),
    )
    nfr_norm = _prf(
        _pred_category_set(predicted_specs, "non_functional_requirements", normalized=True),
        _gold_category_set(samples, "non_functional_requirements", normalized=True),
    )
    con_norm = _prf(
        _pred_category_set(predicted_specs, "constraints", normalized=True),
        _gold_category_set(samples, "constraints", normalized=True),
    )
    avg_stage_4_open_question_count = (
        mean(stage4_open_question_values) if stage4_open_question_values else 0.0
    )
    avg_stage_5_follow_up_count = mean(stage5_followup_values) if stage5_followup_values else 0.0
    if not repair_denominator:
        repair_denominator = float(requirement_quality_metrics.get("requirement_count", 0.0))
    repair_trigger_rate = repair_trigger_total / repair_denominator if repair_denominator else 0.0
    repair_success_rate = (
        repair_success_total / repair_trigger_total if repair_trigger_total else 0.0
    )

    return {
        "sample_count": len(samples),
        "functional_precision": fr_metrics["precision"],
        "functional_recall": fr_metrics["recall"],
        "functional_f1": fr_metrics["f1"],
        "semantic_functional_precision": fr_semantic["precision"],
        "semantic_functional_recall": fr_semantic["recall"],
        "semantic_functional_f1": fr_semantic["f1"],
        "non_functional_precision": nfr_metrics["precision"],
        "non_functional_recall": nfr_metrics["recall"],
        "non_functional_f1": nfr_metrics["f1"],
        "semantic_non_functional_precision": nfr_semantic["precision"],
        "semantic_non_functional_recall": nfr_semantic["recall"],
        "semantic_non_functional_f1": nfr_semantic["f1"],
        "constraint_precision": con_metrics["precision"],
        "constraint_recall": con_metrics["recall"],
        "constraint_f1": con_metrics["f1"],
        "semantic_constraint_precision": con_semantic["precision"],
        "semantic_constraint_recall": con_semantic["recall"],
        "semantic_constraint_f1": con_semantic["f1"],
        "requirement_type_macro_f1": macro_f1,
        "semantic_requirement_macro_f1": semantic_requirement_macro_f1,
        "open_question_recall": oq_metrics["recall"],
        "semantic_open_question_recall": open_question_semantic["recall"],
        "semantic_open_question_f1": open_question_semantic["f1"],
        "follow_up_question_coverage": fu_metrics["recall"],
        "semantic_follow_up_question_coverage": follow_up_semantic["recall"],
        "semantic_follow_up_question_f1": follow_up_semantic["f1"],
        "hallucination_rate": hallucination_rate,
        "exact_hallucination_rate": exact_hallucination_rate,
        **requirement_quality_metrics,
        "schema_json_parse_validity_rate": parse_rate,
        "schema_pydantic_validity_rate": validation_rate,
        "schema_validity_rate": full_schema_rate,
        "json_parse_success_rate": parse_rate,
        "pydantic_validation_success_rate": validation_rate,
        "retry_success_rate": retry_success_rate,
        "retry_recovery_rate": retry_recovery_rate,
        "fallback_rescue_rate": fallback_rescue_rate,
        "repair_trigger_rate": repair_trigger_rate,
        "repair_success_rate": repair_success_rate,
        "final_usable_output_rate": usable_output_rate,
        "semantic_warning_rate": semantic_warning_rate,
        "avg_stage_retry_count": (
            stage_retry_total / stage_retry_observed if stage_retry_observed else 0.0
        ),
        "avg_stage_4_open_question_count": avg_stage_4_open_question_count,
        "avg_stage_5_follow_up_count": avg_stage_5_follow_up_count,
        "stage_failure_counts": stage_failure_counts,
        "constraint_semantic_warning_count": constraint_warning_count,
        "avg_latency_sec": avg_latency,
        "latency_seconds": avg_latency,
        "num_llm_calls": mean(num_llm_calls_values) if num_llm_calls_values else 0.0,
        "functional_f1_normalized": fr_norm["f1"],
        "non_functional_f1_normalized": nfr_norm["f1"],
        "constraint_f1_normalized": con_norm["f1"],
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "N/A"


def _write_sample_debug_artifacts(sample_debug_dir: Path, run: Any) -> None:
    ensure_dir(sample_debug_dir)
    attempt_logs = list(getattr(run, "attempt_logs", []) or [])
    for attempt in attempt_logs:
        idx = int(attempt.get("attempt_index", 0))
        stage = str(attempt.get("stage", "stage"))
        stage_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", stage).strip("._-") or "stage"
        prefix = f"{stage_slug}_attempt_{idx:02d}"
        write_text_file(
            sample_debug_dir / f"{prefix}_raw.txt",
            str(attempt.get("raw_output", "")),
        )
        repaired = attempt.get("repaired_output")
        if repaired:
            write_text_file(
                sample_debug_dir / f"{prefix}_repaired.txt",
                str(repaired),
            )
        error_message = str(attempt.get("error_message") or "").strip()
        if error_message:
            write_text_file(sample_debug_dir / f"{prefix}_error.txt", error_message)

    summary = {
        "status": getattr(run, "status", ""),
        "success": bool(getattr(run, "success", False)),
        "retry_count": int(getattr(run, "retry_count", 0)),
        "json_parse_ok": bool(getattr(run, "json_parse_ok", False)),
        "pydantic_validation_ok": bool(getattr(run, "pydantic_validation_ok", False)),
        "semantic_warnings": list(getattr(run, "semantic_warnings", []) or []),
        "error_message": getattr(run, "error_message", None),
        "stage_stats": getattr(run, "stage_stats", {}),
        "pipeline_mode": getattr(run, "pipeline_mode", "single_shot"),
        "robustness_profile": getattr(run, "robustness_profile", None),
        "latency_sec": float(getattr(run, "latency_sec", 0.0)),
    }
    write_json_file(sample_debug_dir / "summary.json", summary)


def evaluate_model(
    model_label: str,
    pipeline: ConversationToSpecPipeline,
    samples: list[dict[str, Any]],
    output_dir: Path,
    progress_reporter: Any | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reporter = progress_reporter or NullProgressReporter()
    predictions_dir = output_dir / "predictions"
    debug_dir = output_dir / "debug"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    predicted_specs: dict[str, SpecOutput] = {}
    statuses: dict[str, SampleStatus] = {}
    details: list[dict[str, Any]] = []
    reporter.message(f"Evaluating {model_label} on {len(samples)} samples.")

    for sample_index, sample in enumerate(samples, start=1):
        sid = str(sample["id"])
        text = str(sample["conversation_text"])
        sample_debug_dir = debug_dir / sid
        reporter.sample_started(
            sample_index=sample_index,
            total_samples=len(samples),
            sample_id=sid,
        )
        try:
            with reporter.sample_scope(
                sample_index=sample_index,
                total_samples=len(samples),
                sample_id=sid,
            ):
                run = pipeline.run_text(
                    conversation_text=text,
                    progress_reporter=reporter,
                )
            _write_sample_debug_artifacts(sample_debug_dir, run)
            write_text_file(predictions_dir / f"{sid}_raw.txt", run.raw_output)

            if run.success:
                predicted_specs[sid] = run.spec
                write_json_file(predictions_dir / f"{sid}_pred.json", model_dump_compat(run.spec))
                if getattr(run, "verification_report", None):
                    write_json_file(
                        predictions_dir / f"{sid}_verification_report.json",
                        run.verification_report,
                    )
                    write_text_file(
                        predictions_dir / f"{sid}_verification_report.md",
                        format_verification_report_markdown(run.verification_report),
                    )
            else:
                write_text_file(
                    predictions_dir / f"{sid}_error.txt",
                    run.error_message or "Invalid structured output.",
                )

            fallback_used = any(
                key.endswith("_fallback_used") and bool(value)
                for key, value in dict(run.stage_stats).items()
            )
            retry_recovery = bool(run.success and run.retry_count > 0 and not fallback_used)
            fallback_rescue = bool(run.success and fallback_used)
            stage_stats = dict(run.stage_stats)
            semantic_warning = bool(
                len(run.semantic_warnings) > 0
                and stage_stats.get("semantic_verify_enabled", True)
            )
            statuses[sid] = SampleStatus(
                sample_id=sid,
                success=bool(run.success),
                json_parse_ok=bool(run.json_parse_ok),
                pydantic_validation_ok=bool(run.pydantic_validation_ok),
                latency_sec=float(run.latency_sec),
                final_status=str(run.status),
                retry_count=int(run.retry_count),
                retry_success=bool(run.success and run.retry_count > 0),
                retry_recovery=retry_recovery,
                fallback_rescue=fallback_rescue,
                semantic_warning=semantic_warning,
                stage_failure=run.stage_failure,
                stage_retry_counts=dict(run.stage_retry_counts),
                stage_stats=stage_stats,
                error=run.error_message,
            )
            details.append(
                {
                    "sample_id": sid,
                    "success": bool(run.success),
                    "status": str(run.status),
                    "retry_count": int(run.retry_count),
                    "retry_recovery": retry_recovery,
                    "fallback_rescue": fallback_rescue,
                    "stage_retry_counts": dict(run.stage_retry_counts),
                    "stage_failure": run.stage_failure,
                    "stage_stats": stage_stats,
                    "semantic_warning": semantic_warning,
                    "latency_sec": float(run.latency_sec),
                    "json_parse_ok": bool(run.json_parse_ok),
                    "pydantic_validation_ok": bool(run.pydantic_validation_ok),
                    "pipeline_mode": getattr(run, "pipeline_mode", "single_shot"),
                    "robustness_profile": getattr(run, "robustness_profile", None),
                    "error": run.error_message,
                }
            )
            reporter.sample_finished(
                sample_index=sample_index,
                total_samples=len(samples),
                sample_id=sid,
                status=str(run.status),
                latency_sec=float(run.latency_sec),
            )
        except Exception as exc:
            statuses[sid] = SampleStatus(
                sample_id=sid,
                success=False,
                json_parse_ok=False,
                pydantic_validation_ok=False,
                latency_sec=0.0,
                final_status="failed_invalid_output",
                retry_count=0,
                retry_success=False,
                retry_recovery=False,
                fallback_rescue=False,
                semantic_warning=False,
                stage_failure="exception",
                stage_retry_counts={},
                stage_stats={},
                error=str(exc),
            )
            write_text_file(predictions_dir / f"{sid}_error.txt", str(exc))
            details.append(
                {
                    "sample_id": sid,
                    "success": False,
                    "status": "failed_invalid_output",
                    "retry_count": 0,
                    "retry_recovery": False,
                    "fallback_rescue": False,
                    "stage_retry_counts": {},
                    "stage_failure": "exception",
                    "stage_stats": {},
                    "semantic_warning": False,
                    "latency_sec": 0.0,
                    "json_parse_ok": False,
                    "pydantic_validation_ok": False,
                    "pipeline_mode": getattr(pipeline, "pipeline_mode", "single_shot"),
                    "robustness_profile": getattr(pipeline, "robustness_profile", None),
                    "error": str(exc),
                }
            )
            reporter.sample_finished(
                sample_index=sample_index,
                total_samples=len(samples),
                sample_id=sid,
                status="failed_invalid_output",
                latency_sec=0.0,
            )

    metrics = compute_metrics(samples, predicted_specs, statuses)
    if not getattr(pipeline, "_semantic_verify_enabled", lambda: True)():
        metrics["semantic_warning_rate"] = None
    write_json_file(output_dir / "metrics.json", metrics)
    report = {
        "model": model_label,
        "metrics": metrics,
        "samples": details,
        "run_metadata": run_metadata or {},
    }
    if run_metadata:
        write_json_file(output_dir / "run_config.json", run_metadata)
    write_json_file(output_dir / "details.json", report)
    return report


def build_comparison_table(results: dict[str, dict[str, Any]]) -> str:
    columns = [
        "model",
        "functional_f1",
        "semantic_functional_f1",
        "non_functional_f1",
        "semantic_non_functional_f1",
        "constraint_f1",
        "semantic_constraint_f1",
        "macro_f1",
        "semantic_req_macro_f1",
        "open_question_recall",
        "semantic_open_question_recall",
        "follow_up_coverage",
        "semantic_follow_up_coverage",
        "hallucination_rate",
        "requirement_count",
        "acceptance_criteria_coverage",
        "evidence_span_coverage",
        "groundedness_rate",
        "unsupported_requirement_rate",
        "verification_pass_rate",
        "traceability_coverage",
        "quality_gate_pass_rate",
        "high_ambiguity_rate",
        "schema_validity_rate",
        "json_parse_success",
        "pydantic_success",
        "retry_success",
        "retry_recovery",
        "fallback_rescue",
        "repair_trigger_rate",
        "repair_success_rate",
        "usable_output",
        "semantic_warning",
        "stage4_open_questions",
        "stage5_follow_ups",
        "num_llm_calls",
        "avg_latency_sec",
    ]
    header = "| " + " | ".join(columns) + " |\n"
    divider = "|" + "|".join(["---", *["---:"] * (len(columns) - 1)]) + "|\n"
    rows: list[str] = []
    for model_label, report in results.items():
        metrics = report.get("metrics", {})
        rows.append(
            "| "
            + " | ".join(
                [
                    model_label,
                    _format_metric(metrics.get('functional_f1', 0.0)),
                    _format_metric(metrics.get('semantic_functional_f1', 0.0)),
                    _format_metric(metrics.get('non_functional_f1', 0.0)),
                    _format_metric(metrics.get('semantic_non_functional_f1', 0.0)),
                    _format_metric(metrics.get('constraint_f1', 0.0)),
                    _format_metric(metrics.get('semantic_constraint_f1', 0.0)),
                    _format_metric(metrics.get('requirement_type_macro_f1', 0.0)),
                    _format_metric(metrics.get('semantic_requirement_macro_f1', 0.0)),
                    _format_metric(metrics.get('open_question_recall', 0.0)),
                    _format_metric(metrics.get('semantic_open_question_recall', 0.0)),
                    _format_metric(metrics.get('follow_up_question_coverage', 0.0)),
                    _format_metric(metrics.get('semantic_follow_up_question_coverage', 0.0)),
                    _format_metric(metrics.get('hallucination_rate', 0.0)),
                    _format_metric(metrics.get('requirement_count', 0.0)),
                    _format_metric(metrics.get('acceptance_criteria_coverage', 0.0)),
                    _format_metric(metrics.get('evidence_span_coverage', 0.0)),
                    _format_metric(metrics.get('groundedness_rate', 0.0)),
                    _format_metric(metrics.get('unsupported_requirement_rate', 0.0)),
                    _format_metric(metrics.get('verification_pass_rate', 0.0)),
                    _format_metric(metrics.get('traceability_coverage', 0.0)),
                    _format_metric(metrics.get('quality_gate_pass_rate', 0.0)),
                    _format_metric(metrics.get('high_ambiguity_rate', 0.0)),
                    _format_metric(metrics.get('schema_validity_rate', 0.0)),
                    _format_metric(metrics.get('json_parse_success_rate', 0.0)),
                    _format_metric(metrics.get('pydantic_validation_success_rate', 0.0)),
                    _format_metric(metrics.get('retry_success_rate', 0.0)),
                    _format_metric(metrics.get('retry_recovery_rate', 0.0)),
                    _format_metric(metrics.get('fallback_rescue_rate', 0.0)),
                    _format_metric(metrics.get('repair_trigger_rate', 0.0)),
                    _format_metric(metrics.get('repair_success_rate', 0.0)),
                    _format_metric(metrics.get('final_usable_output_rate', 0.0)),
                    _format_metric(metrics.get('semantic_warning_rate', 0.0)),
                    _format_metric(metrics.get('avg_stage_4_open_question_count', 0.0)),
                    _format_metric(metrics.get('avg_stage_5_follow_up_count', 0.0)),
                    _format_metric(metrics.get('num_llm_calls', 0.0)),
                    _format_metric(metrics.get('avg_latency_sec', 0.0)),
                ]
            )
            + " |"
        )
    return header + divider + "\n".join(rows) + ("\n" if rows else "")
