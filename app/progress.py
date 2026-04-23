from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, TextIO


STAGE_0 = "stage_0_segmentation"
STAGE_FINAL = "final_assembly"

PIPELINE_PROGRESS_SEQUENCE = (
    STAGE_0,
    "stage_1_candidate_extraction",
    "stage_2_candidate_classification",
    "stage_3_requirement_rewriting",
    "stage_4_open_question_generation",
    "stage_5_followup_generation",
    "stage_6_project_summary",
    STAGE_FINAL,
)

PIPELINE_STAGE_LABELS = {
    STAGE_0: "Segment conversation",
    "stage_1_candidate_extraction": "Extract candidates",
    "stage_2_candidate_classification": "Classify candidates",
    "stage_3_requirement_rewriting": "Rewrite requirements",
    "stage_4_open_question_generation": "Generate open questions",
    "stage_5_followup_generation": "Generate follow-up questions",
    "stage_6_project_summary": "Summarize project",
    STAGE_FINAL: "Assemble final spec",
}


def pipeline_total_steps() -> int:
    return len(PIPELINE_PROGRESS_SEQUENCE)


def pipeline_step_index(stage_key: str) -> int:
    try:
        return PIPELINE_PROGRESS_SEQUENCE.index(stage_key) + 1
    except ValueError:
        return len(PIPELINE_PROGRESS_SEQUENCE)


def stage_display_name(stage_key: str) -> str:
    return PIPELINE_STAGE_LABELS.get(stage_key, stage_key.replace("_", " ").strip().title())


@dataclass
class _NullAttemptHandle:
    def finish(self, result_text: str) -> None:
        return None


class NullProgressReporter:
    def pipeline_started(self, *, total_steps: int, run_label: str | None = None) -> None:
        return None

    def pipeline_finished(self, *, status: str, elapsed_sec: float) -> None:
        return None

    def stage_started(self, *, stage_key: str, step_index: int, total_steps: int) -> None:
        return None

    def stage_finished(
        self,
        *,
        stage_key: str,
        step_index: int,
        total_steps: int,
        result_text: str,
    ) -> None:
        return None

    def stage_attempt_started(
        self,
        *,
        stage_key: str,
        step_index: int,
        total_steps: int,
        attempt_index: int,
        max_attempts: int,
    ) -> _NullAttemptHandle:
        return _NullAttemptHandle()

    def message(self, text: str) -> None:
        return None

    def sample_started(self, *, sample_index: int, total_samples: int, sample_id: str) -> None:
        return None

    def sample_finished(
        self,
        *,
        sample_index: int,
        total_samples: int,
        sample_id: str,
        status: str,
        latency_sec: float,
    ) -> None:
        return None

    @contextmanager
    def sample_scope(
        self,
        *,
        sample_index: int,
        total_samples: int,
        sample_id: str,
    ) -> Iterator["NullProgressReporter"]:
        yield self


@dataclass
class _ConsoleAttemptHandle:
    reporter: "ConsoleProgressReporter"
    stage_key: str
    step_index: int
    total_steps: int
    attempt_index: int
    max_attempts: int
    started_at: float = field(default_factory=time.perf_counter)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    rendered_inline: bool = False

    def start(self) -> "_ConsoleAttemptHandle":
        if not self.reporter.dynamic_updates:
            return self

        self.thread = threading.Thread(
            target=self._run,
            name=f"progress-{self.stage_key}-{self.attempt_index}",
            daemon=True,
        )
        self.thread.start()
        return self

    def _run(self) -> None:
        while not self.stop_event.wait(self.reporter.heartbeat_interval_sec):
            elapsed = time.perf_counter() - self.started_at
            if elapsed < self.reporter.heartbeat_delay_sec:
                continue
            self.rendered_inline = True
            self.reporter._render_inline(
                self._running_text(elapsed_sec=int(elapsed))
            )

    def _running_text(self, *, elapsed_sec: int) -> str:
        return (
            f"{self.reporter._stage_prefix(self.step_index, self.total_steps)} "
            f"{stage_display_name(self.stage_key)} running "
            f"(attempt {self.attempt_index}/{self.max_attempts})... {elapsed_sec}s elapsed"
        )

    def finish(self, result_text: str) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.reporter.heartbeat_interval_sec * 2 + 0.1)

        elapsed_sec = time.perf_counter() - self.started_at
        final_text = (
            f"{self.reporter._stage_prefix(self.step_index, self.total_steps)} "
            f"{stage_display_name(self.stage_key)} {result_text} "
            f"(attempt {self.attempt_index}/{self.max_attempts}, {elapsed_sec:.1f}s)"
        )
        self.reporter._finalize_attempt(self, final_text)


class ConsoleProgressReporter:
    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        dynamic_updates: bool | None = None,
        heartbeat_interval_sec: float = 1.0,
        heartbeat_delay_sec: float = 1.0,
    ) -> None:
        self.stream = stream or sys.stdout
        if dynamic_updates is None:
            dynamic_updates = bool(getattr(self.stream, "isatty", lambda: False)())
        self.dynamic_updates = bool(dynamic_updates)
        self.heartbeat_interval_sec = max(0.01, float(heartbeat_interval_sec))
        self.heartbeat_delay_sec = max(0.0, float(heartbeat_delay_sec))
        self._lock = threading.Lock()
        self._inline_width = 0
        self._active_attempt: _ConsoleAttemptHandle | None = None
        self._sample_prefix = ""

    def _write_line(self, text: str) -> None:
        with self._lock:
            if self._inline_width:
                self.stream.write("\n")
                self._inline_width = 0
            self.stream.write(text + "\n")
            self.stream.flush()

    def _render_inline(self, text: str) -> None:
        with self._lock:
            width = max(self._inline_width, len(text))
            self.stream.write("\r" + text.ljust(width))
            self.stream.flush()
            self._inline_width = width

    def _finalize_attempt(self, handle: _ConsoleAttemptHandle, text: str) -> None:
        with self._lock:
            if self.dynamic_updates and handle.rendered_inline:
                width = max(self._inline_width, len(text))
                self.stream.write("\r" + text.ljust(width) + "\n")
            else:
                if self._inline_width:
                    self.stream.write("\n")
                self.stream.write(text + "\n")
            self.stream.flush()
            self._inline_width = 0
            if self._active_attempt is handle:
                self._active_attempt = None

    def _stage_prefix(self, step_index: int, total_steps: int) -> str:
        percent = int((step_index / max(1, total_steps)) * 100)
        return f"{self._sample_prefix}[{step_index}/{total_steps} {percent:>3}%]"

    def pipeline_started(self, *, total_steps: int, run_label: str | None = None) -> None:
        label_suffix = f" for {run_label}" if run_label else ""
        self._write_line(f"{self._sample_prefix}Pipeline started{label_suffix} ({total_steps} steps).")

    def pipeline_finished(self, *, status: str, elapsed_sec: float) -> None:
        self._write_line(
            f"{self._sample_prefix}Pipeline finished with status={status} in {elapsed_sec:.1f}s."
        )

    def stage_started(self, *, stage_key: str, step_index: int, total_steps: int) -> None:
        self._write_line(
            f"{self._stage_prefix(step_index, total_steps)} {stage_display_name(stage_key)} started"
        )

    def stage_finished(
        self,
        *,
        stage_key: str,
        step_index: int,
        total_steps: int,
        result_text: str,
    ) -> None:
        self._write_line(
            f"{self._stage_prefix(step_index, total_steps)} "
            f"{stage_display_name(stage_key)} {result_text}"
        )

    def stage_attempt_started(
        self,
        *,
        stage_key: str,
        step_index: int,
        total_steps: int,
        attempt_index: int,
        max_attempts: int,
    ) -> _ConsoleAttemptHandle:
        handle = _ConsoleAttemptHandle(
            reporter=self,
            stage_key=stage_key,
            step_index=step_index,
            total_steps=total_steps,
            attempt_index=attempt_index,
            max_attempts=max_attempts,
        ).start()
        self._active_attempt = handle
        return handle

    def message(self, text: str) -> None:
        self._write_line(text)

    def sample_started(self, *, sample_index: int, total_samples: int, sample_id: str) -> None:
        self._write_line(
            f"Sample {sample_index}/{total_samples} [{sample_id}] started"
        )

    def sample_finished(
        self,
        *,
        sample_index: int,
        total_samples: int,
        sample_id: str,
        status: str,
        latency_sec: float,
    ) -> None:
        self._write_line(
            f"Sample {sample_index}/{total_samples} [{sample_id}] finished "
            f"with status={status} in {latency_sec:.1f}s"
        )

    @contextmanager
    def sample_scope(
        self,
        *,
        sample_index: int,
        total_samples: int,
        sample_id: str,
    ) -> Iterator["ConsoleProgressReporter"]:
        previous_prefix = self._sample_prefix
        self._sample_prefix = f"[sample {sample_index}/{total_samples} {sample_id}] "
        try:
            yield self
        finally:
            self._sample_prefix = previous_prefix
