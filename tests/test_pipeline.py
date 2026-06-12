import json
from pathlib import Path

from app.evaluation import evaluate_model
from app.pipeline import ConversationToSpecPipeline
from app.prompt_builder import load_prompt_config
from tests.fakes import FakeSingleShotRunner


def _sample_text() -> str:
    return (
        "We need a cafe website with menu and hours.\n"
        "Customers should reserve tables online.\n"
        "Online payment is not part of the first release.\n"
        "The design should feel clean and modern."
    )


def _build_pipeline(runner=None, generation_config=None) -> ConversationToSpecPipeline:
    return ConversationToSpecPipeline(
        runner=runner or FakeSingleShotRunner(),
        prompt_config=load_prompt_config(),
        generation_config=generation_config or {"max_retries": 0},
        prompt_style="few_shot",
        verify_mode="heuristic",
    )


def test_single_shot_pipeline_preserves_schema(tmp_path: Path):
    input_path = tmp_path / "sample.txt"
    input_path.write_text(_sample_text(), encoding="utf-8")
    pipeline = _build_pipeline()
    run = pipeline.run_file(input_path=input_path, output_dir=tmp_path)

    assert run.success is True
    assert run.pipeline_mode == "single_shot"
    assert run.num_llm_calls >= 1
    assert run.spec.project_summary
    assert run.spec.conversation_units
    assert (tmp_path / "spec.json").exists()
    assert (tmp_path / "spec.md").exists()
    assert (tmp_path / "verification_report.json").exists()
    loaded = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert "functional_requirements" in loaded
    assert loaded["functional_requirements"][0]["acceptance_criteria"]
    assert loaded["functional_requirements"][0]["quality_checks"]
    report = json.loads((tmp_path / "verification_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["num_llm_calls"] == run.num_llm_calls
    assert report["summary"]["generator_llm_calls"] == 1
    assert report["summary"]["verifier_llm_calls"] == run.num_llm_calls - 1


def test_chain_mode_is_removed():
    try:
        ConversationToSpecPipeline(
            runner=FakeSingleShotRunner(),
            prompt_config=load_prompt_config(),
            pipeline_mode="chain",
        )
    except ValueError as exc:
        assert "chain mode has been removed" in str(exc)
    else:
        raise AssertionError("chain mode should not be accepted")


def test_invalid_single_shot_output_writes_error_artifacts(tmp_path: Path):
    input_path = tmp_path / "sample.txt"
    input_path.write_text(_sample_text(), encoding="utf-8")
    pipeline = _build_pipeline(runner=FakeSingleShotRunner(raw_output="not-json"))
    run = pipeline.run_file(input_path=input_path, output_dir=tmp_path)

    assert run.success is False
    assert run.stage_failure == "single_shot_spec_generation"
    assert (tmp_path / "error.log").exists()
    assert (tmp_path / "debug" / "spec" / "summary.json").exists()
    assert (tmp_path / "debug" / "spec" / "single_shot_spec_generation_attempt_01_error.txt").exists()


def test_debug_artifacts_cleanup_removes_stale_files(tmp_path: Path):
    output_dir = tmp_path / "out"
    stale_debug_dir = output_dir / "debug" / "spec"
    stale_debug_dir.mkdir(parents=True, exist_ok=True)
    stale_file = stale_debug_dir / "stale_from_old_run.txt"
    stale_file.write_text("stale", encoding="utf-8")
    stale_error_log = output_dir / "error.log"
    stale_error_log.write_text("old error", encoding="utf-8")

    input_path = tmp_path / "sample.txt"
    input_path.write_text(_sample_text(), encoding="utf-8")
    pipeline = _build_pipeline()
    run = pipeline.run_file(input_path=input_path, output_dir=output_dir)

    assert run.success is True
    assert run.debug_dir is not None
    assert not stale_file.exists()
    assert not stale_error_log.exists()
    assert (run.debug_dir / "summary.json").exists()


def test_stage_failure_does_not_crash_batch_evaluation(tmp_path: Path):
    samples = [
        {
            "id": "s1",
            "conversation_text": _sample_text(),
            "gold": {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "constraints": [],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
            },
        },
        {
            "id": "s2",
            "conversation_text": "Need scheduling app.\nUsers should receive reminders.",
            "gold": {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "constraints": [],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
            },
        },
    ]
    pipeline = _build_pipeline(runner=FakeSingleShotRunner(raw_output="not-json"))
    report = evaluate_model("fake-fail", pipeline, samples, tmp_path)
    assert report["metrics"]["sample_count"] == 2
    assert (tmp_path / "predictions" / "s1_error.txt").exists()
    assert (tmp_path / "predictions" / "s2_error.txt").exists()
