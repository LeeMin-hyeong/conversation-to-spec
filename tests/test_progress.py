import io
import time
from pathlib import Path

from app.evaluation import evaluate_model
from app.model_runner import MockModelRunner
from app.pipeline import ConversationToSpecPipeline
from app.progress import ConsoleProgressReporter, pipeline_step_index, pipeline_total_steps
from app.prompt_builder import load_prompt_config


def _build_pipeline() -> ConversationToSpecPipeline:
    return ConversationToSpecPipeline(
        runner=MockModelRunner(),
        prompt_config=load_prompt_config(),
        generation_config={"max_retries": 1},
    )


def test_console_progress_reporter_updates_elapsed_time_inline():
    stream = io.StringIO()
    reporter = ConsoleProgressReporter(
        stream=stream,
        dynamic_updates=True,
        heartbeat_interval_sec=0.01,
        heartbeat_delay_sec=0.0,
    )
    total_steps = pipeline_total_steps()
    step_index = pipeline_step_index("stage_1_candidate_extraction")

    reporter.pipeline_started(total_steps=total_steps, run_label="demo")
    reporter.stage_started(
        stage_key="stage_1_candidate_extraction",
        step_index=step_index,
        total_steps=total_steps,
    )
    handle = reporter.stage_attempt_started(
        stage_key="stage_1_candidate_extraction",
        step_index=step_index,
        total_steps=total_steps,
        attempt_index=1,
        max_attempts=2,
    )
    time.sleep(0.05)
    handle.finish("completed")
    reporter.pipeline_finished(status="success", elapsed_sec=0.05)

    output = stream.getvalue()
    assert "\r" in output
    assert "Extract candidates running" in output
    assert "Extract candidates completed" in output


def test_pipeline_reports_stage_progress_messages():
    stream = io.StringIO()
    reporter = ConsoleProgressReporter(stream=stream, dynamic_updates=False)
    pipeline = _build_pipeline()

    run = pipeline.run_text(
        conversation_text="Customers should reserve tables online.\nThe site should be fast on mobile.",
        progress_reporter=reporter,
    )

    output = stream.getvalue()
    assert run.success is True
    assert "Pipeline started" in output
    assert "Segment conversation completed" in output
    assert "Extract candidates" in output
    assert "Assemble final spec completed" in output


def test_evaluate_model_reports_sample_progress(tmp_path: Path):
    samples = [
        {
            "id": "mini",
            "conversation_text": "Need scheduling tool.\nIt should be fast.",
            "gold": {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
                "constraints": [],
            },
        }
    ]
    stream = io.StringIO()
    reporter = ConsoleProgressReporter(stream=stream, dynamic_updates=False)
    pipeline = _build_pipeline()

    report = evaluate_model(
        model_label="mock",
        pipeline=pipeline,
        samples=samples,
        output_dir=tmp_path,
        progress_reporter=reporter,
    )

    output = stream.getvalue()
    assert "metrics" in report
    assert "Evaluating mock on 1 samples." in output
    assert "Sample 1/1 [mini] started" in output
    assert "Sample 1/1 [mini] finished" in output
