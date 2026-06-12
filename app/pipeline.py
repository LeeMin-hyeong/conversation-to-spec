from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.extractor import ExtractionMeta, extract_spec_output_safe, semantic_verify
from app.formatter import format_spec_markdown
from app.model_runner import BaseModelRunner
from app.parser import load_conversation_text
from app.postprocessor import confidence_aware_postprocess
from app.progress import (
    NullProgressReporter,
    STAGE_0,
    STAGE_MODEL_PREP,
    STAGE_VERIFICATION,
    pipeline_step_index,
    pipeline_total_steps,
)
from app.prompt_builder import build_single_shot_spec_prompt
from app.quality import ensure_spec_quality_defaults, validate_spec_quality
from app.schemas import ConversationUnit, SpecOutput
from app.segmenter import segment_conversation
from app.utils import ensure_dir, model_dump_compat, write_json_file, write_text_file
from app.verifier import SpecVerifier, format_verification_report_markdown


PIPELINE_STATUS_SUCCESS = "success"
PIPELINE_STATUS_REPAIRED_SUCCESS = "repaired_success"
PIPELINE_STATUS_SEMANTIC_WARNING = "semantic_warning"
PIPELINE_STATUS_FAILED_INVALID_OUTPUT = "failed_invalid_output"

STAGE_SINGLE_SHOT = "single_shot_spec_generation"

# Kept as an empty compatibility constant for old imports. The legacy
# multi-stage implementation has been removed from the executable path.
ROBUSTNESS_PROFILE_CONFIGS: dict[str, dict[str, Any]] = {}


@dataclass
class PipelineRunResult:
    spec: SpecOutput
    raw_output: str
    extraction_meta: ExtractionMeta
    latency_sec: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    success: bool
    status: str
    retry_count: int
    error_message: Optional[str] = None
    attempt_logs: list[dict[str, Any]] = field(default_factory=list)
    stage_retry_counts: dict[str, int] = field(default_factory=dict)
    stage_failure: Optional[str] = None
    stage_stats: dict[str, Any] = field(default_factory=dict)
    pipeline_mode: str = "single_shot"
    robustness_profile: Optional[str] = None
    output_json_path: Optional[Path] = None
    output_md_path: Optional[Path] = None
    verification_report_json_path: Optional[Path] = None
    verification_report_md_path: Optional[Path] = None
    verification_report: dict[str, Any] = field(default_factory=dict)
    num_llm_calls: int = 0
    generation_info: dict[str, Any] = field(default_factory=dict)
    debug_dir: Optional[Path] = None

    @property
    def json_parse_ok(self) -> bool:
        return self.extraction_meta.json_parse_ok

    @property
    def pydantic_validation_ok(self) -> bool:
        return self.extraction_meta.pydantic_validation_ok

    @property
    def semantic_warnings(self) -> list[str]:
        return list(self.extraction_meta.semantic_warnings)


class ConversationToSpecPipeline:
    def __init__(
        self,
        runner: BaseModelRunner,
        prompt_config: dict,
        generation_config: Optional[dict] = None,
        pipeline_mode: str = "single_shot",
        robustness_profile: Optional[str] = None,
        robustness_config: Optional[dict[str, Any]] = None,
        prompt_style: str = "few_shot",
        verify_mode: str = "minicheck",
        repair_on_fail: bool = False,
    ) -> None:
        if pipeline_mode != "single_shot":
            raise ValueError("chain mode has been removed; use pipeline_mode='single_shot'.")
        if prompt_style not in {"zero_shot", "few_shot"}:
            raise ValueError("prompt_style must be 'zero_shot' or 'few_shot'.")
        if verify_mode not in {"off", "heuristic", "llm", "minicheck"}:
            raise ValueError("verify_mode must be one of: off, heuristic, llm, minicheck.")

        self.runner = runner
        self.prompt_config = {**prompt_config, "prompt_style": prompt_style}
        self.generation_config = generation_config or {}
        self.pipeline_mode = "single_shot"
        self.prompt_style = prompt_style
        self.verify_mode = verify_mode
        self.repair_on_fail = repair_on_fail
        self.robustness_profile = robustness_profile
        self.robustness_config = robustness_config or {}
        self.progress_reporter = NullProgressReporter()

    def _repair_enabled(self) -> bool:
        return bool(self.robustness_config.get("repair_enabled", True))

    def _semantic_verify_enabled(self) -> bool:
        return bool(self.robustness_config.get("semantic_verify_enabled", True))

    @staticmethod
    def _short_progress_error(error_message: str, max_len: int = 96) -> str:
        collapsed = re.sub(r"\s+", " ", str(error_message or "").strip())
        if len(collapsed) <= max_len:
            return collapsed
        return collapsed[: max_len - 3] + "..."

    @staticmethod
    def _build_fallback_spec(conversation_units: list[Any], error_message: str) -> SpecOutput:
        return SpecOutput(
            project_summary=(
                "The single-call pipeline could not produce a valid specification. "
                "Please inspect debug artifacts and retry with a more stable prompt or model."
            ),
            functional_requirements=[],
            non_functional_requirements=[],
            constraints=[],
            open_questions=[],
            follow_up_questions=[],
            notes=[],
            conversation_units=conversation_units,
            verification_warnings=[error_message],
        )

    def _run_spec_verifier(
        self,
        spec: SpecOutput,
        conversation_units: list[Any],
    ) -> tuple[SpecOutput, dict[str, Any], str, list[str], int]:
        verifier = SpecVerifier(
            runner=self.runner,
            generation_config=self.generation_config,
        )
        result = verifier.run(
            spec,
            conversation_units,
            verify_mode=self.verify_mode,
            repair_on_fail=self.repair_on_fail,
        )
        verified_spec = result.spec
        if result.warnings:
            verified_spec.verification_warnings = sorted(
                set(list(verified_spec.verification_warnings) + result.warnings)
            )
        return (
            verified_spec,
            result.report,
            result.report_markdown,
            result.warnings,
            result.num_llm_calls,
        )

    @staticmethod
    def _requirement_items(spec: SpecOutput) -> list[Any]:
        return (
            list(spec.functional_requirements)
            + list(spec.non_functional_requirements)
            + list(spec.constraints)
        )

    def _stage_stats(
        self,
        spec: SpecOutput,
        *,
        verifier_report: dict[str, Any],
        verifier_num_llm_calls: int,
    ) -> dict[str, Any]:
        items = self._requirement_items(spec)
        stats: dict[str, Any] = {
            "pipeline_mode": "single_shot",
            "repair_enabled": self._repair_enabled(),
            "semantic_verify_enabled": self._semantic_verify_enabled(),
            "prompt_style": self.prompt_style,
            "verify_mode": self.verify_mode,
            "repair_on_fail": self.repair_on_fail,
            "num_llm_calls": 1 + verifier_num_llm_calls,
            "verifier_llm_calls": verifier_num_llm_calls,
            "functional_requirement_count": len(spec.functional_requirements),
            "non_functional_requirement_count": len(spec.non_functional_requirements),
            "constraint_count": len(spec.constraints),
            "stage_4_open_question_count": len(spec.open_questions),
            "stage_5_follow_up_count": len(spec.follow_up_questions),
            "requirement_quality_enrichment_count": len(items),
            "requirement_quality_acceptance_criteria_count": sum(
                len(item.acceptance_criteria) for item in items
            ),
            "requirement_quality_evidence_span_count": sum(
                len(item.evidence_spans) for item in items
            ),
        }
        if verifier_report:
            summary = verifier_report.get("summary", {})
            stats["verification_groundedness_rate"] = summary.get("groundedness_rate", 0.0)
            stats["verification_pass_rate"] = summary.get("verification_pass_rate", 0.0)
            stats["repair_trigger_count"] = summary.get("repair_trigger_count", 0)
            stats["repair_success_count"] = summary.get("repair_success_count", 0)
        return stats

    @staticmethod
    def _augment_verification_report_calls(
        report: dict[str, Any],
        *,
        generator_llm_calls: int,
        verifier_llm_calls: int,
    ) -> dict[str, Any]:
        if not report:
            return report
        summary = report.setdefault("summary", {})
        summary["generator_llm_calls"] = generator_llm_calls
        summary["verifier_llm_calls"] = verifier_llm_calls
        summary["num_llm_calls"] = generator_llm_calls + verifier_llm_calls
        return report

    @staticmethod
    def _write_debug_artifacts(debug_dir: Path, run: PipelineRunResult) -> None:
        if debug_dir.exists():
            for child in debug_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        ensure_dir(debug_dir)
        for attempt in run.attempt_logs:
            stage = str(attempt.get("stage", "stage"))
            idx = int(attempt.get("attempt_index", 0))
            stage_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", stage).strip("._-") or "stage"
            prefix = f"{stage_slug}_attempt_{idx:02d}"
            write_text_file(debug_dir / f"{prefix}_raw.txt", str(attempt.get("raw_output", "")))
            repaired = attempt.get("repaired_output")
            if repaired:
                write_text_file(debug_dir / f"{prefix}_repaired.txt", str(repaired))
            error_message = str(attempt.get("error_message") or "").strip()
            if error_message:
                write_text_file(debug_dir / f"{prefix}_error.txt", error_message)

        summary = {
            "status": run.status,
            "success": run.success,
            "retry_count": run.retry_count,
            "json_parse_ok": run.json_parse_ok,
            "pydantic_validation_ok": run.pydantic_validation_ok,
            "used_repair": run.extraction_meta.used_repair,
            "semantic_warnings": run.semantic_warnings,
            "error_message": run.error_message,
            "latency_sec": run.latency_sec,
            "generation_latency_sec": run.generation_info.get("latency_sec"),
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "generation_info": run.generation_info,
            "attempt_count": len(run.attempt_logs),
            "stage_retry_counts": run.stage_retry_counts,
            "stage_failure": run.stage_failure,
            "stage_stats": run.stage_stats,
            "pipeline_mode": run.pipeline_mode,
            "robustness_profile": run.robustness_profile,
            "num_llm_calls": run.num_llm_calls,
            "verification_report_summary": (run.verification_report or {}).get("summary", {}),
        }
        write_json_file(debug_dir / "summary.json", summary)

    def run_text(
        self,
        conversation_text: str,
        output_dir: Optional[Path] = None,
        output_basename: str = "spec",
        save_markdown: bool = True,
        progress_reporter: Any | None = None,
    ) -> PipelineRunResult:
        started = time.perf_counter()
        previous_reporter = self.progress_reporter
        self.progress_reporter = progress_reporter or NullProgressReporter()
        total_steps = pipeline_total_steps()
        raw_output = ""
        verification_report: dict[str, Any] = {}
        verification_report_md = ""

        try:
            self.progress_reporter.pipeline_started(total_steps=total_steps)
            try:
                self.progress_reporter.stage_started(
                    stage_key=STAGE_0,
                    step_index=1,
                    total_steps=total_steps,
                )
                conversation_units = segment_conversation(conversation_text)
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_0,
                    step_index=1,
                    total_steps=total_steps,
                    result_text=f"completed ({len(conversation_units)} units)",
                )
            except Exception as exc:
                conversation_units = [
                    ConversationUnit(id="U1", text=conversation_text.strip() or "(empty conversation)")
                ]
                error_message = f"Conversation segmentation failed: {exc}"
                meta = ExtractionMeta(validation_error=error_message)
                run = PipelineRunResult(
                    spec=self._build_fallback_spec(conversation_units, error_message),
                    raw_output="",
                    extraction_meta=meta,
                    latency_sec=time.perf_counter() - started,
                    prompt_tokens=None,
                    completion_tokens=None,
                    success=False,
                    status=PIPELINE_STATUS_FAILED_INVALID_OUTPUT,
                    retry_count=0,
                    error_message=error_message,
                    stage_failure=STAGE_0,
                    stage_stats={"pipeline_mode": "single_shot", "num_llm_calls": 0},
                    num_llm_calls=0,
                )
                self._persist_outputs(
                    run,
                    output_dir=output_dir,
                    output_basename=output_basename,
                    save_markdown=save_markdown,
                    verification_report_md=verification_report_md,
                )
                self.progress_reporter.pipeline_finished(status=run.status, elapsed_sec=run.latency_sec)
                return run

            self.progress_reporter.stage_started(
                stage_key=STAGE_MODEL_PREP,
                step_index=pipeline_step_index(STAGE_MODEL_PREP),
                total_steps=total_steps,
            )
            try:
                self.runner.prepare()
            except Exception as exc:
                error_message = str(exc)
                meta = ExtractionMeta(
                    json_parse_ok=False,
                    pydantic_validation_ok=False,
                    validation_error=error_message,
                )
                spec = self._build_fallback_spec(conversation_units, error_message)
                run = PipelineRunResult(
                    spec=spec,
                    raw_output="",
                    extraction_meta=meta,
                    latency_sec=time.perf_counter() - started,
                    prompt_tokens=None,
                    completion_tokens=None,
                    success=False,
                    status=PIPELINE_STATUS_FAILED_INVALID_OUTPUT,
                    retry_count=0,
                    error_message=error_message,
                    stage_failure=STAGE_MODEL_PREP,
                    stage_stats={"pipeline_mode": "single_shot", "num_llm_calls": 0},
                    num_llm_calls=0,
                )
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_MODEL_PREP,
                    step_index=pipeline_step_index(STAGE_MODEL_PREP),
                    total_steps=total_steps,
                    result_text=f"failed: {self._short_progress_error(error_message)}",
                )
                self._persist_outputs(
                    run,
                    output_dir=output_dir,
                    output_basename=output_basename,
                    save_markdown=save_markdown,
                    verification_report_md="",
                )
                self.progress_reporter.pipeline_finished(status=run.status, elapsed_sec=run.latency_sec)
                return run
            self.progress_reporter.stage_finished(
                stage_key=STAGE_MODEL_PREP,
                step_index=pipeline_step_index(STAGE_MODEL_PREP),
                total_steps=total_steps,
                result_text="completed",
            )

            self.progress_reporter.stage_started(
                stage_key=STAGE_SINGLE_SHOT,
                step_index=pipeline_step_index(STAGE_SINGLE_SHOT),
                total_steps=total_steps,
            )
            prompt = build_single_shot_spec_prompt(
                conversation_units,
                self.prompt_config,
                prompt_style=self.prompt_style,
            )
            try:
                raw_output = self.runner.generate(prompt, self.generation_config)
            except Exception as exc:
                error_message = str(exc)
                meta = ExtractionMeta(
                    json_parse_ok=False,
                    pydantic_validation_ok=False,
                    validation_error=error_message,
                )
                spec = self._build_fallback_spec(conversation_units, error_message)
                run = PipelineRunResult(
                    spec=spec,
                    raw_output="",
                    extraction_meta=meta,
                    latency_sec=time.perf_counter() - started,
                    prompt_tokens=None,
                    completion_tokens=None,
                    success=False,
                    status=PIPELINE_STATUS_FAILED_INVALID_OUTPUT,
                    retry_count=0,
                    error_message=error_message,
                    attempt_logs=[
                        {
                            "stage": STAGE_SINGLE_SHOT,
                            "attempt_index": 1,
                            "prompt_type": "initial",
                            "raw_output": "",
                            "repaired_output": None,
                            "json_parse_ok": False,
                            "pydantic_validation_ok": False,
                            "used_repair": False,
                            "error_message": error_message,
                        }
                    ],
                    stage_failure=STAGE_SINGLE_SHOT,
                    stage_stats={"pipeline_mode": "single_shot", "num_llm_calls": 1},
                    num_llm_calls=1,
                )
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_SINGLE_SHOT,
                    step_index=pipeline_step_index(STAGE_SINGLE_SHOT),
                    total_steps=total_steps,
                    result_text=f"failed: {self._short_progress_error(error_message)}",
                )
                self._persist_outputs(
                    run,
                    output_dir=output_dir,
                    output_basename=output_basename,
                    save_markdown=save_markdown,
                    verification_report_md="",
                )
                self.progress_reporter.pipeline_finished(status=run.status, elapsed_sec=run.latency_sec)
                return run
            info = self.runner.last_generation_info or {}
            prompt_tokens = info.get("prompt_tokens")
            completion_tokens = info.get("completion_tokens")

            spec, meta = extract_spec_output_safe(
                raw_output,
                conversation_units,
                allow_repair=self._repair_enabled(),
            )
            semantic_warnings: list[str] = []
            verifier_num_llm_calls = 0
            error_message: Optional[str] = None
            stage_failure: Optional[str] = None
            success = spec is not None

            if spec is None:
                error_message = meta.validation_error or meta.parse_error or "Invalid single-shot output."
                stage_failure = STAGE_SINGLE_SHOT
                spec = self._build_fallback_spec(conversation_units, error_message)
                status = PIPELINE_STATUS_FAILED_INVALID_OUTPUT
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_SINGLE_SHOT,
                    step_index=pipeline_step_index(STAGE_SINGLE_SHOT),
                    total_steps=total_steps,
                    result_text=f"failed: {self._short_progress_error(error_message)}",
                )
            else:
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_SINGLE_SHOT,
                    step_index=pipeline_step_index(STAGE_SINGLE_SHOT),
                    total_steps=total_steps,
                    result_text="completed",
                )
                self.progress_reporter.stage_started(
                    stage_key=STAGE_VERIFICATION,
                    step_index=pipeline_step_index(STAGE_VERIFICATION),
                    total_steps=total_steps,
                )
                if self._semantic_verify_enabled():
                    spec, semantic_warnings = semantic_verify(spec, conversation_units)
                spec = ensure_spec_quality_defaults(spec, conversation_units)
                quality_warnings = validate_spec_quality(spec, conversation_units)
                semantic_warnings = sorted(set(semantic_warnings + quality_warnings))
                spec.verification_warnings = sorted(
                    set(list(spec.verification_warnings) + quality_warnings)
                )
                spec, verification_report, verification_report_md, verifier_warnings, verifier_num_llm_calls = (
                    self._run_spec_verifier(spec, conversation_units)
                )
                postprocess_warnings: list[str] = []
                if self.verify_mode != "off":
                    postprocess_result = confidence_aware_postprocess(
                        spec,
                        conversation_units,
                        runner=self.runner,
                        generation_config=self.generation_config,
                    )
                    verifier_num_llm_calls += postprocess_result.num_llm_calls
                    if postprocess_result.warnings:
                        postprocess_warnings = postprocess_result.warnings
                        spec = postprocess_result.spec
                        spec.verification_warnings = sorted(
                            set(list(spec.verification_warnings) + postprocess_warnings)
                        )
                    if postprocess_result.changed:
                        (
                            spec,
                            verification_report,
                            verification_report_md,
                            verifier_warnings,
                            extra_verifier_calls,
                        ) = self._run_spec_verifier(spec, conversation_units)
                        verifier_num_llm_calls += extra_verifier_calls
                verification_report = self._augment_verification_report_calls(
                    verification_report,
                    generator_llm_calls=1,
                    verifier_llm_calls=verifier_num_llm_calls,
                )
                verification_report_md = format_verification_report_markdown(
                    verification_report
                )
                if verifier_warnings:
                    semantic_warnings = sorted(set(semantic_warnings + verifier_warnings))
                if postprocess_warnings:
                    semantic_warnings = sorted(set(semantic_warnings + postprocess_warnings))
                meta.semantic_warnings = semantic_warnings
                if semantic_warnings:
                    status = PIPELINE_STATUS_SEMANTIC_WARNING
                elif meta.used_repair:
                    status = PIPELINE_STATUS_REPAIRED_SUCCESS
                else:
                    status = PIPELINE_STATUS_SUCCESS
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_VERIFICATION,
                    step_index=pipeline_step_index(STAGE_VERIFICATION),
                    total_steps=total_steps,
                    result_text="completed",
                )

            stage_stats = self._stage_stats(
                spec,
                verifier_report=verification_report,
                verifier_num_llm_calls=verifier_num_llm_calls,
            )
            attempt_logs = [
                {
                    "stage": STAGE_SINGLE_SHOT,
                    "attempt_index": 1,
                    "prompt_type": "initial",
                    "raw_output": raw_output,
                    "repaired_output": meta.repaired_output,
                    "json_parse_ok": meta.json_parse_ok,
                    "pydantic_validation_ok": meta.pydantic_validation_ok,
                    "used_repair": meta.used_repair,
                    "error_message": error_message,
                }
            ]
            latency_sec = time.perf_counter() - started
            run = PipelineRunResult(
                spec=spec,
                raw_output=raw_output,
                extraction_meta=meta,
                latency_sec=latency_sec,
                prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
                success=success,
                status=status,
                retry_count=0,
                error_message=error_message,
                attempt_logs=attempt_logs,
                stage_retry_counts={},
                stage_failure=stage_failure,
                stage_stats=stage_stats,
                pipeline_mode="single_shot",
                robustness_profile=self.robustness_profile,
                verification_report=verification_report,
                num_llm_calls=stage_stats["num_llm_calls"],
                generation_info=info,
            )

            self._persist_outputs(
                run,
                output_dir=output_dir,
                output_basename=output_basename,
                save_markdown=save_markdown,
                verification_report_md=verification_report_md,
            )
            self.progress_reporter.pipeline_finished(status=run.status, elapsed_sec=run.latency_sec)
            return run
        finally:
            self.progress_reporter = previous_reporter

    def _persist_outputs(
        self,
        run: PipelineRunResult,
        *,
        output_dir: Optional[Path],
        output_basename: str,
        save_markdown: bool,
        verification_report_md: str,
    ) -> None:
        if output_dir is None:
            return
        ensure_dir(output_dir)
        output_json_path = output_dir / f"{output_basename}.json"
        output_md_path = output_dir / f"{output_basename}.md"
        verification_report_json_path = output_dir / "verification_report.json"
        verification_report_md_path = output_dir / "verification_report.md"
        write_json_file(output_json_path, model_dump_compat(run.spec))
        write_json_file(verification_report_json_path, run.verification_report)
        if save_markdown:
            write_text_file(output_md_path, format_spec_markdown(run.spec))
            write_text_file(verification_report_md_path, verification_report_md)
        error_log_path = output_dir / "error.log"
        if not run.success:
            write_text_file(error_log_path, run.error_message or "Invalid structured output.")
        elif error_log_path.exists():
            error_log_path.unlink()
        debug_dir = output_dir / "debug" / output_basename
        self._write_debug_artifacts(debug_dir, run)
        run.output_json_path = output_json_path
        run.output_md_path = output_md_path
        run.verification_report_json_path = verification_report_json_path
        run.verification_report_md_path = verification_report_md_path
        run.debug_dir = debug_dir

    def run_file(
        self,
        input_path: Path,
        output_dir: Path,
        output_basename: str = "spec",
        progress_reporter: Any | None = None,
    ) -> PipelineRunResult:
        text = load_conversation_text(input_path)
        return self.run_text(
            conversation_text=text,
            output_dir=output_dir,
            output_basename=output_basename,
            save_markdown=True,
            progress_reporter=progress_reporter,
        )
