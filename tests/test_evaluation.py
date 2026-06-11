from pathlib import Path

from app.evaluation import build_comparison_table, compute_metrics, evaluate_model
from app.pipeline import ConversationToSpecPipeline
from app.prompt_builder import load_prompt_config
from app.schemas import (
    ConstraintItem,
    ConversationUnit,
    RequirementItem,
    RequirementQualityChecks,
    RequirementVerification,
    SpecOutput,
)
from tests.fakes import FakeSingleShotRunner


def test_compute_metrics_includes_single_shot_quality_keys():
    samples = [
        {
            "id": "s1",
            "conversation_text": "Need booking.",
            "gold": {
                "functional_requirements": [{"text": "The system shall provide booking."}],
                "non_functional_requirements": [],
                "constraints": [{"text": "Online payment shall not be included in v1."}],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
            },
        }
    ]
    predicted = {
        "s1": SpecOutput(
            project_summary="Summary.",
            functional_requirements=[
                RequirementItem(
                    id="FR1",
                    text="The system shall provide booking.",
                    source_units=["U1"],
                    evidence_spans=["Need booking."],
                    acceptance_criteria=[
                        "Given booking is in scope, When a user books, Then the booking is saved."
                    ],
                    quality_checks=RequirementQualityChecks(
                        is_atomic=True,
                        is_testable=True,
                        has_clear_actor=True,
                        has_traceable_evidence=True,
                        ambiguity_risk="low",
                    ),
                    verification=RequirementVerification(
                        source_relevance_score=0.9,
                        verdict="SUPPORTED",
                        confidence=0.9,
                        warnings=[],
                    ),
                )
            ],
            non_functional_requirements=[],
            constraints=[
                ConstraintItem(
                    id="CON1",
                    text="Online payment shall not be included in v1.",
                    source_units=["U1"],
                    evidence_spans=["Need booking."],
                    acceptance_criteria=[
                        "Given v1 scope is reviewed, When payment scope is checked, Then online payment is excluded."
                    ],
                    quality_checks=RequirementQualityChecks(
                        is_atomic=True,
                        is_testable=True,
                        has_clear_actor=True,
                        has_traceable_evidence=True,
                        ambiguity_risk="low",
                    ),
                    verification=RequirementVerification(
                        source_relevance_score=0.8,
                        verdict="SUPPORTED",
                        confidence=0.8,
                        warnings=[],
                    ),
                )
            ],
            open_questions=[],
            follow_up_questions=[],
            notes=[],
            conversation_units=[ConversationUnit(id="U1", text="Need booking.")],
            verification_warnings=[],
        )
    }
    statuses = {
        "s1": type(
            "Status",
            (),
            {
                "sample_id": "s1",
                "success": True,
                "json_parse_ok": True,
                "pydantic_validation_ok": True,
                "latency_sec": 0.5,
                "final_status": "success",
                "retry_count": 1,
                "retry_success": True,
                "retry_recovery": True,
                "fallback_rescue": False,
                "semantic_warning": False,
                "stage_failure": None,
                "stage_retry_counts": {"stage_2_candidate_classification": 1},
                "stage_stats": {
                    "stage_1_candidate_count": 5,
                    "stage_2_discard_rate": 0.4,
                    "stage_4_open_question_count": 2,
                    "stage_5_follow_up_count": 3,
                },
            },
        )()
    }
    metrics = compute_metrics(samples, predicted, statuses)
    assert metrics["functional_f1"] == 1.0
    assert metrics["semantic_functional_f1"] == 1.0
    assert metrics["constraint_f1"] == 1.0
    assert metrics["semantic_constraint_f1"] == 1.0
    assert "semantic_open_question_recall" in metrics
    assert "semantic_follow_up_question_coverage" in metrics
    assert "semantic_requirement_macro_f1" in metrics
    assert "json_parse_success_rate" in metrics
    assert "pydantic_validation_success_rate" in metrics
    assert "retry_success_rate" in metrics
    assert "retry_recovery_rate" in metrics
    assert "fallback_rescue_rate" in metrics
    assert "final_usable_output_rate" in metrics
    assert "semantic_warning_rate" in metrics
    assert "avg_stage_1_candidate_count" in metrics
    assert "avg_stage_2_discard_rate" in metrics
    assert "avg_stage_4_open_question_count" in metrics
    assert "avg_stage_5_follow_up_count" in metrics
    assert metrics["acceptance_criteria_coverage"] == 1.0
    assert metrics["evidence_span_coverage"] == 1.0
    assert metrics["traceability_coverage"] == 1.0
    assert metrics["quality_gate_pass_rate"] == 1.0
    assert metrics["high_ambiguity_rate"] == 0.0
    assert metrics["requirement_count"] == 2.0
    assert metrics["source_relevance_avg"] == 0.8500000000000001
    assert metrics["groundedness_rate"] == 1.0
    assert metrics["unsupported_requirement_rate"] == 0.0
    assert metrics["verification_pass_rate"] == 1.0
    assert "repair_trigger_rate" in metrics
    assert "repair_success_rate" in metrics
    assert "num_llm_calls" in metrics
    assert "stage_failure_counts" in metrics
    assert "constraint_semantic_warning_count" in metrics
    assert metrics["avg_stage_4_open_question_count"] == 2.0
    assert metrics["avg_stage_5_follow_up_count"] == 3.0


def test_compute_metrics_accepts_legacy_stage_key_fallback():
    samples = [
        {
            "id": "s1",
            "conversation_text": "Need booking.",
            "gold": {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "constraints": [],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
            },
        }
    ]
    predicted = {}
    statuses = {
        "s1": type(
            "Status",
            (),
            {
                "sample_id": "s1",
                "success": False,
                "json_parse_ok": False,
                "pydantic_validation_ok": False,
                "latency_sec": 0.0,
                "final_status": "failed",
                "retry_count": 0,
                "retry_success": False,
                "retry_recovery": False,
                "fallback_rescue": False,
                "semantic_warning": False,
                "stage_failure": None,
                "stage_retry_counts": {},
                "stage_stats": {"stage_4_follow_up_count": 4},
            },
        )()
    }
    metrics = compute_metrics(samples, predicted, statuses)
    assert metrics["avg_stage_4_open_question_count"] == 4.0
    assert metrics["avg_stage_5_follow_up_count"] == 4.0
    assert metrics["avg_stage_4_follow_up_count"] == 4.0


def test_comparison_table_uses_new_stage_diagnostic_columns():
    table = build_comparison_table(
        {
            "fake": {
                "metrics": {
                    "functional_f1": 1.0,
                    "semantic_functional_f1": 1.0,
                    "non_functional_f1": 0.5,
                    "semantic_non_functional_f1": 0.5,
                    "constraint_f1": 0.25,
                    "semantic_constraint_f1": 0.25,
                    "requirement_type_macro_f1": 0.75,
                    "semantic_requirement_macro_f1": 0.6,
                    "open_question_recall": 0.1,
                    "follow_up_question_coverage": 0.2,
                    "hallucination_rate": 0.3,
                    "acceptance_criteria_coverage": 0.31,
                    "evidence_span_coverage": 0.32,
                    "traceability_coverage": 0.33,
                    "quality_gate_pass_rate": 0.34,
                    "high_ambiguity_rate": 0.35,
                    "schema_validity_rate": 0.4,
                    "json_parse_success_rate": 0.5,
                    "pydantic_validation_success_rate": 0.6,
                    "retry_success_rate": 0.7,
                    "retry_recovery_rate": 0.65,
                    "fallback_rescue_rate": 0.15,
                    "final_usable_output_rate": 0.8,
                    "semantic_warning_rate": 0.9,
                    "avg_stage_1_candidate_count": 1.1,
                    "avg_stage_2_discard_rate": 1.2,
                    "avg_stage_4_open_question_count": 1.3,
                    "avg_stage_5_follow_up_count": 1.4,
                    "avg_latency_sec": 1.5,
                }
            }
        }
    )
    assert "stage4_open_questions" in table
    assert "stage5_follow_ups" in table
    assert "semantic_req_macro_f1" in table
    assert "retry_recovery" in table
    assert "fallback_rescue" in table
    assert "acceptance_criteria_coverage" in table
    assert "quality_gate_pass_rate" in table
    assert "0.3100" in table
    assert "0.3400" in table
    assert "1.3000" in table
    assert "1.4000" in table


def test_comparison_table_accepts_legacy_stage_metric_key():
    table = build_comparison_table(
        {
            "legacy": {
                "metrics": {
                    "avg_stage_4_follow_up_count": 2.5,
                }
            }
        }
    )
    assert "stage4_open_questions" in table
    assert "stage5_follow_ups" in table
    assert "2.5000 | 2.5000" in table


def test_comparison_table_marks_null_semantic_warning_as_na():
    table = build_comparison_table(
        {
            "NoSemanticVerify": {
                "metrics": {
                    "semantic_warning_rate": None,
                }
            }
        }
    )
    assert "N/A" in table


def test_evaluate_model_writes_debug_and_prediction_artifacts(tmp_path: Path):
    samples = [
        {
            "id": "mini",
            "conversation_text": "Need scheduling tool.\nIt should be fast.\nMaybe payment later.",
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
    pipeline = ConversationToSpecPipeline(
        runner=FakeSingleShotRunner(),
        prompt_config=load_prompt_config(),
        generation_config={},
        verify_mode="heuristic",
    )
    report = evaluate_model(
        model_label="fake",
        pipeline=pipeline,
        samples=samples,
        output_dir=tmp_path,
        run_metadata={"pipeline_mode": "single_shot"},
    )
    assert "metrics" in report
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "run_config.json").exists()
    assert (tmp_path / "predictions" / "mini_pred.json").exists()
    assert (tmp_path / "debug" / "mini" / "summary.json").exists()
