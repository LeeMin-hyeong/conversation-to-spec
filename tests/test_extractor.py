import json

from app.extractor import (
    build_stage_1_fallback_candidates,
    ExtractionError,
    extract_spec_output,
    extract_spec_output_safe,
    semantic_verify,
    validate_stage_1_candidates,
    validate_stage_2_classification,
    validate_stage_3_rewriting,
    validate_requirement_quality_enrichment,
    validate_stage_4_open_questions,
    validate_stage_5_followups,
)
from app.schemas import (
    CandidateItem,
    ConstraintItem,
    ConversationUnit,
    NoteItem,
    QuestionItem,
    RequirementItem,
    SpecOutput,
    Stage1CandidatesOutput,
    Stage3RewrittenOutput,
    RewrittenItem,
)


def _units() -> list[ConversationUnit]:
    return [
        ConversationUnit(id="U1", text="We may add online payment later."),
        ConversationUnit(id="U2", text="The app should be fast on mobile."),
    ]


def test_extractor_repairs_json_with_code_fence():
    raw = """```json
{
  "project_summary": "Summary.",
  "functional_requirements": [
    {"id":"FR1","text":"The system shall support booking.","source_units":["U1"],}
  ],
  "non_functional_requirements": [],
  "constraints": [],
  "open_questions": [],
  "follow_up_questions": [],
  "notes": []
}
```"""
    spec, meta = extract_spec_output(raw, _units())
    assert spec.functional_requirements[0].id == "FR1"
    assert meta.used_repair is True


def test_extractor_repairs_json_with_extra_text():
    raw = """
Model answer:
Here is the JSON:
{
  "project_summary": "Summary.",
  "functional_requirements": [
    {"id":"FR1","text":"The system shall support booking.","source_units":["U1"]}
  ],
  "non_functional_requirements": [],
  "constraints": [],
  "open_questions": [],
  "follow_up_questions": [],
  "notes": []
}
Thanks.
"""
    spec, meta = extract_spec_output(raw, _units())
    assert spec.project_summary == "Summary."
    assert meta.json_parse_ok is True


def test_extractor_strips_qwen_thinking_before_json_parse():
    raw = """
<think>
I might mention {"not": "the answer"} while reasoning.
</think>
{
  "project_summary": "Summary.",
  "functional_requirements": [
    {"id":"FR1","text":"The system shall support booking.","source_units":["U1"]}
  ],
  "non_functional_requirements": [],
  "constraints": [],
  "open_questions": [],
  "follow_up_questions": [],
  "notes": []
}
"""
    spec, meta = extract_spec_output(raw, _units())
    assert spec.project_summary == "Summary."
    assert meta.json_parse_ok is True


def test_extractor_raises_when_schema_missing_required_field():
    raw = """{
  "functional_requirements": [],
  "non_functional_requirements": [],
  "constraints": [],
  "open_questions": [],
  "follow_up_questions": [],
  "notes": []
}"""
    try:
        extract_spec_output(raw, _units())
        assert False, "Expected ExtractionError"
    except ExtractionError as exc:
        assert "Missing required top-level fields" in str(exc)


def test_extractor_accepts_source_unit_decision_schema():
    units = [
        ConversationUnit(id="U1", text="Customers should be able to reserve tables online."),
        ConversationUnit(id="U2", text="The site should load quickly on mobile."),
        ConversationUnit(id="U3", text="We may add online payment later."),
        ConversationUnit(id="U4", text="The design should feel clean and modern."),
    ]
    raw = json.dumps(
        {
            "project_summary": "Cafe booking site.",
            "source_unit_decisions": [
                {"source_unit": "U1", "decision": "functional_requirement", "claim": ""},
                {"source_unit": "U2", "decision": "non_functional_requirement", "claim": ""},
                {"source_unit": "U3", "decision": "constraint", "claim": ""},
                {"source_unit": "U4", "decision": "open_question", "claim": ""},
            ],
        }
    )
    spec, meta = extract_spec_output_safe(raw, units)
    assert meta.pydantic_validation_ok is True
    assert len(spec.functional_requirements) == 1
    assert "reserve tables online" in spec.functional_requirements[0].text
    assert len(spec.non_functional_requirements) == 1
    assert "load quickly on mobile" in spec.non_functional_requirements[0].text
    assert len(spec.constraints) == 1
    assert "Online payment shall be deferred" in spec.constraints[0].text
    assert any(question.source_units == ["U3"] for question in spec.open_questions)
    assert any(question.source_units == ["U4"] for question in spec.open_questions)
    assert "source_unit_decision_schema_used" in spec.verification_warnings


def test_source_unit_decision_schema_prefers_linguistic_signals_over_model_decision():
    units = [
        ConversationUnit(
            id="U1",
            text="Most staff use tablets in the shop, so the dashboard should work well on tablet screens.",
        ),
        ConversationUnit(id="U2", text="We need the first version ready before the autumn fair."),
        ConversationUnit(id="U3", text="The owner says the interface should feel calm and professional."),
        ConversationUnit(
            id="U4",
            text="We are not sure whether customers should create accounts or use one-time links.",
        ),
    ]
    raw = json.dumps(
        {
            "project_summary": "Repair shop app.",
            "source_unit_decisions": [
                {"source_unit": "U1", "decision": "constraint"},
                {"source_unit": "U2", "decision": "functional_requirement"},
                {"source_unit": "U3", "decision": "note"},
                {"source_unit": "U4", "decision": "functional_requirement"},
            ],
        }
    )

    spec, meta = extract_spec_output_safe(raw, units)

    assert meta.pydantic_validation_ok is True
    assert any("tablet screen" in item.text.lower() for item in spec.non_functional_requirements)
    assert any("autumn fair" in item.text.lower() for item in spec.constraints)
    assert not any("create accounts" in item.text.lower() for item in spec.functional_requirements)
    assert any("create accounts or use one-time links" in item.text.lower() for item in spec.open_questions)
    assert any("calm and professional" in item.text.lower() for item in spec.open_questions)


def test_source_unit_decision_schema_normalizes_explicit_launch_exclusion():
    units = [
        ConversationUnit(
            id="U1",
            text="Online rent payment may be added later, but it should not be part of launch.",
        ),
        ConversationUnit(
            id="U2",
            text="We are still deciding whether residents should create accounts or use invite links.",
        ),
    ]
    raw = json.dumps(
        {
            "project_summary": "Resident portal.",
            "source_unit_decisions": [
                {"source_unit": "U1", "decision": "constraint"},
                {"source_unit": "U2", "decision": "open_question"},
            ],
        }
    )

    spec, meta = extract_spec_output_safe(raw, units)

    assert meta.pydantic_validation_ok is True
    assert len(spec.constraints) == 1
    assert spec.constraints[0].text == "Online rent payment shall be excluded from the launch scope."
    assert not any("included in the first release or deferred" in q.text for q in spec.open_questions)
    assert any("Should residents create accounts or use invite links?" == q.text for q in spec.open_questions)


def test_source_unit_decision_schema_accepts_multiple_atomic_decisions():
    units = [
        ConversationUnit(
            id="U1",
            text=(
                "Applicants should upload site plans and signed owner consent forms, "
                "but legal validity checking is not part of the first release."
            ),
        ),
        ConversationUnit(
            id="U2",
            text=(
                "Staff should see who changed an application status and when, "
                "although detailed legal audit exports can wait until phase two."
            ),
        ),
    ]
    raw = json.dumps(
        {
            "project_summary": "Permit portal.",
            "source_unit_decisions": [
                {
                    "source_unit": "U1",
                    "atomic_decisions": [
                        {
                            "decision": "functional_requirement",
                            "claim": "Applicants can upload site plans and signed owner consent forms.",
                        },
                        {
                            "decision": "constraint",
                            "claim": "Legal validity checking is not part of the first release.",
                        },
                    ],
                },
                {
                    "source_unit": "U2",
                    "atomic_decisions": [
                        {
                            "decision": "functional_requirement",
                            "claim": "Staff can see who changed an application status and when.",
                        },
                        {
                            "decision": "constraint",
                            "claim": "Detailed legal audit exports can wait until phase two.",
                        },
                    ],
                },
            ],
        }
    )

    spec, meta = extract_spec_output_safe(raw, units)

    assert meta.pydantic_validation_ok is True
    assert len(spec.functional_requirements) == 2
    assert any("applicants to upload site plans" in item.text.lower() for item in spec.functional_requirements)
    assert any("staff to see who changed" in item.text.lower() for item in spec.functional_requirements)
    assert len(spec.constraints) == 2
    assert any("legal validity checking" in item.text.lower() for item in spec.constraints)
    assert any("detailed legal audit exports" in item.text.lower() for item in spec.constraints)
    assert all(item.source_units in (["U1"], ["U2"]) for item in [*spec.functional_requirements, *spec.constraints])


def test_repeated_source_unit_decision_rows_create_multiple_atomic_outputs():
    units = [
        ConversationUnit(
            id="U1",
            text=(
                "Applicants should upload site plans and signed owner consent forms, "
                "but legal validity checking is not part of the first release."
            ),
        ),
    ]
    raw = json.dumps(
        {
            "project_summary": "Permit portal.",
            "source_unit_decisions": [
                {
                    "source_unit": "U1",
                    "decision": "functional_requirement",
                    "claim": "Applicants should upload site plans and signed owner consent forms.",
                },
                {
                    "source_unit": "U1",
                    "decision": "constraint",
                    "claim": "Legal validity checking is not part of the first release.",
                },
            ],
        }
    )

    spec, meta = extract_spec_output_safe(raw, units)

    assert meta.pydantic_validation_ok is True
    assert len(spec.functional_requirements) == 1
    assert len(spec.constraints) == 1
    assert "applicants to upload site plans" in spec.functional_requirements[0].text.lower()
    assert "legal validity checking" in spec.constraints[0].text.lower()
    assert spec.functional_requirements[0].source_units == ["U1"]
    assert spec.constraints[0].source_units == ["U1"]


def test_blank_claim_source_unit_decision_is_split_into_atomic_clauses():
    units = [
        ConversationUnit(
            id="U1",
            text=(
                "Applicants should be able to start one permit application, upload site plans, "
                "contractor license files, and signed owner consent forms, but the first release "
                "only needs to check that required documents are present, not whether the documents "
                "are legally valid."
            ),
        ),
        ConversationUnit(
            id="U2",
            text=(
                "For audit reasons, staff should be able to see who changed an application status "
                "and when, although detailed legal audit exports can wait until phase two."
            ),
        ),
    ]
    raw = json.dumps(
        {
            "project_summary": "Permit portal.",
            "source_unit_decisions": [
                {"source_unit": "U1", "decision": "functional_requirement", "claim": ""},
                {"source_unit": "U2", "decision": "constraint", "claim": ""},
            ],
        }
    )

    spec, meta = extract_spec_output_safe(raw, units)

    assert meta.pydantic_validation_ok is True
    assert len(spec.functional_requirements) == 2
    assert len(spec.constraints) == 2
    assert any("upload site plans" in item.text.lower() for item in spec.functional_requirements)
    assert any("see who changed" in item.text.lower() for item in spec.functional_requirements)
    assert any("first release" in item.text.lower() for item in spec.constraints)
    assert any("detailed legal audit exports" in item.text.lower() for item in spec.constraints)
    assert any("split_blank_claim_into_2_atomic_claims" in warning for warning in spec.verification_warnings)


def test_semantic_verify_uses_atomic_evidence_not_full_source_for_future_scope():
    units = [
        ConversationUnit(
            id="U1",
            text=(
                "For audit reasons, staff should be able to see who changed an application status "
                "and when, although detailed legal audit exports can wait until phase two."
            ),
        ),
    ]
    raw = json.dumps(
        {
            "project_summary": "Permit portal.",
            "source_unit_decisions": [
                {"source_unit": "U1", "decision": "constraint", "claim": ""},
            ],
        }
    )

    spec, meta = extract_spec_output_safe(raw, units)
    verified, warnings = semantic_verify(spec, units)

    assert meta.pydantic_validation_ok is True
    assert any("staff to see who changed" in item.text.lower() for item in verified.functional_requirements)
    assert any("detailed legal audit exports" in item.text.lower() for item in verified.constraints)
    assert not any("staff should be able to see" in item.text.lower() for item in verified.constraints)
    audit_constraint = next(item for item in verified.constraints if "detailed legal audit exports" in item.text.lower())
    assert audit_constraint.evidence_spans == ["detailed legal audit exports can wait until phase two"]
    assert not any("future_scope)" in warning for warning in warnings if "raw_functional_requirements" in warning)


def test_semantic_verify_downgrades_future_scope_requirement():
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[
            RequirementItem(
                id="FR1",
                text="The system shall support online payment.",
                source_units=["U1"],
            )
        ],
        non_functional_requirements=[],
        constraints=[],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=_units(),
    )
    verified, warnings = semantic_verify(spec, _units())
    assert not verified.functional_requirements
    assert any("Future-scope candidate requirement" in note.text for note in verified.notes)
    assert warnings


def test_semantic_verify_flags_unsupported_requirement():
    units = [ConversationUnit(id="U1", text="We need a simple booking page.")]
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[
            RequirementItem(
                id="FR1",
                text="The system shall integrate blockchain settlement for payments.",
                source_units=["U1"],
            )
        ],
        non_functional_requirements=[],
        constraints=[],
        open_questions=[QuestionItem(text="Existing question", source_units=["U1"])],
        follow_up_questions=[],
        notes=[NoteItem(text="Existing note", source_units=["U1"])],
        conversation_units=units,
    )
    verified, warnings = semantic_verify(spec, units)
    assert not verified.functional_requirements
    assert any("Please confirm requirement scope and intent" in q.text for q in verified.open_questions)
    assert any("low_semantic_overlap" in w for w in warnings)


def test_semantic_verify_adds_missing_high_confidence_source_units():
    units = [
        ConversationUnit(
            id="U1",
            text="We need a website where customers can see today's menu and opening hours.",
        ),
        ConversationUnit(
            id="U2",
            text="Customers should be able to reserve tables online.",
        ),
        ConversationUnit(
            id="U3",
            text="Most visitors will use phones, so the site should load quickly on mobile.",
        ),
        ConversationUnit(id="U4", text="The design should feel clean and modern."),
        ConversationUnit(id="U5", text="We may add online payment later."),
    ]
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[
            RequirementItem(
                id="FR1",
                text="The system shall allow customers to reserve tables online.",
                source_units=["U2"],
            )
        ],
        non_functional_requirements=[],
        constraints=[],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=units,
    )

    verified, warnings = semantic_verify(spec, units)

    assert any("today's menu and opening hours" in item.text for item in verified.functional_requirements)
    assert any("load quickly on mobile" in item.text for item in verified.non_functional_requirements)
    assert any(item.source_units == ["U4"] for item in verified.open_questions)
    assert any(item.source_units == ["U5"] for item in verified.notes)
    assert any("coverage_fallback_added_functional_requirement" in warning for warning in warnings)
    assert any("coverage_fallback_added_non_functional_requirement" in warning for warning in warnings)


def test_extract_spec_removes_few_shot_example_contamination_before_semantic_fallback():
    units = [
        ConversationUnit(id="U1", text="Customers should be able to reserve tables online."),
        ConversationUnit(id="U2", text="Staff must be able to update menu items."),
    ]
    raw = {
        "project_summary": "The conversation describes EX_PRODUCT_A.",
        "functional_requirements": [
            {
                "id": "FR1",
                "text": "The system shall allow EX_ACTOR_A to EX_ACTION_A EX_OBJECT_A.",
                "source_units": ["U_EX1"],
                "evidence_spans": ["EX_ACTOR_A should EX_ACTION_A EX_OBJECT_A."],
                "acceptance_criteria": [
                    "Given EX_ACTOR_A starts EX_WORKFLOW_A, When they submit valid input, Then the system shall complete EX_ACTION_A."
                ],
            }
        ],
        "non_functional_requirements": [],
        "constraints": [],
        "open_questions": [{"text": "Is there any ambiguity about the functionality?", "source_units": ["U1"]}],
        "follow_up_questions": [{"text": "Are there any specific developers asking about this?", "source_units": ["U1"]}],
        "notes": [{"text": "Future or contextual note.", "source_units": ["U_EX3"]}],
    }

    spec, meta = extract_spec_output_safe(json.dumps(raw), units)
    assert spec is not None
    assert meta.pydantic_validation_ok is True
    assert not spec.functional_requirements
    assert not spec.open_questions
    assert not spec.follow_up_questions
    assert not spec.notes
    assert any("few_shot_contamination_removed_functional_requirements" in warning for warning in spec.verification_warnings)

    verified, warnings = semantic_verify(spec, units)
    assert any("reserve tables online" in item.text for item in verified.functional_requirements)
    assert any("coverage_fallback_added_functional_requirement" in warning for warning in warnings)
    assert any(
        "few_shot_contamination_removed_functional_requirements" in warning
        for warning in verified.verification_warnings
    )


def test_semantic_verify_downgrades_constraint_without_hard_boundary_signal():
    units = [ConversationUnit(id="U1", text="Staff should have access to the admin page.")]
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[],
        non_functional_requirements=[],
        constraints=[
            ConstraintItem(
                id="CON1",
                text="Staff should have access to the admin page.",
                source_units=["U1"],
            )
        ],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=units,
    )

    verified, warnings = semantic_verify(spec, units)
    assert not verified.constraints
    assert any("staff" in item.text.lower() and "admin page" in item.text.lower() for item in verified.functional_requirements)
    assert any("not_hard_constraint" in warning for warning in warnings)


def test_semantic_verify_normalizes_nfr_text_from_repaired_source():
    units = [
        ConversationUnit(
            id="U1",
            text="Staff must be able to update menu items and prices from an admin page.",
        ),
        ConversationUnit(
            id="U2",
            text="Most visitors will use phones, so the site should load quickly on mobile.",
        ),
    ]
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[],
        non_functional_requirements=[
            RequirementItem(
                id="NFR1",
                text="The system shall ensure most visitors will use phones.",
                source_units=["U1"],
            )
        ],
        constraints=[],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=units,
    )

    verified, warnings = semantic_verify(spec, units)

    assert len(verified.non_functional_requirements) == 1
    assert verified.non_functional_requirements[0].source_units == ["U2"]
    assert "load quickly on mobile" in verified.non_functional_requirements[0].text
    assert any("source_units_repaired_by_inference" in warning for warning in warnings)


def test_semantic_verify_records_future_scope_constraint_and_note():
    units = [ConversationUnit(id="U1", text="We may add online payment later.")]
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[],
        non_functional_requirements=[],
        constraints=[
            ConstraintItem(
                id="CON1",
                text="Online payment may be added later.",
                source_units=["U1"],
            )
        ],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=units,
    )
    verified, warnings = semantic_verify(spec, units)
    assert len(verified.constraints) == 1
    assert "Online payment shall be deferred" in verified.constraints[0].text
    assert not any("Future-scope boundary candidate" in note.text for note in verified.notes)
    assert any("Should online payment" in question.text for question in verified.open_questions)
    assert any("future_scope_without_explicit_boundary" in w for w in warnings)


def test_semantic_verify_adds_constraint_for_explicit_later_phase_scope():
    units = [ConversationUnit(id="U1", text="We might add ID card integration in a later phase.")]
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[],
        non_functional_requirements=[],
        constraints=[],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=units,
    )

    verified, warnings = semantic_verify(spec, units)
    assert len(verified.constraints) == 1
    assert "ID card integration shall be deferred" in verified.constraints[0].text
    assert any("coverage_fallback_added_constraint" in warning for warning in warnings)


def test_semantic_verify_keeps_explicit_release_boundary_constraint():
    units = [ConversationUnit(id="U1", text="Online payment is not part of the first release.")]
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[],
        non_functional_requirements=[],
        constraints=[
            ConstraintItem(
                id="CON1",
                text="Online payment shall not be included in the initial release.",
                source_units=["U1"],
            )
        ],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=units,
    )
    verified, warnings = semantic_verify(spec, units)
    assert len(verified.constraints) == 1
    assert not any("future_scope_without_explicit_boundary" in w for w in warnings)


def test_stage_4_accepts_string_questions_and_infers_sources():
    units = [
        ConversationUnit(id="U1", text="Customers should reserve tables online."),
        ConversationUnit(id="U2", text="The design should feel clean and modern."),
    ]
    out = validate_stage_4_open_questions(
        {
            "open_questions": [
                "What acceptance criteria define clean and modern design?"
            ]
        },
        units,
    )
    assert len(out.open_questions) == 1
    assert out.open_questions[0].text.endswith("?")
    assert out.open_questions[0].source_units == ["U2"]


def test_stage_5_accepts_common_question_field_alias():
    units = [ConversationUnit(id="U1", text="Customers should reserve tables online.")]
    out = validate_stage_5_followups(
        {
            "open_questions": [
                {
                    "question": "Which reservation time slots should be supported?",
                    "source_units": ["U1"],
                }
            ]
        },
        units,
    )
    assert len(out.follow_up_questions) == 1
    assert out.follow_up_questions[0].source_units == ["U1"]


def test_stage_5_accepts_followup_alias_and_multiline_string():
    units = [
        ConversationUnit(id="U1", text="Customers should reserve tables online for specific time slots."),
        ConversationUnit(id="U2", text="The design should feel clean and modern."),
    ]
    out = validate_stage_5_followups(
        {
            "followup_questions": (
                "- Which specific reservation time slots should be supported?\n"
                "- What measurable criteria define clean and modern design?"
            )
        },
        units,
    )
    assert len(out.follow_up_questions) == 2
    assert all(item.text.endswith("?") for item in out.follow_up_questions)
    assert all(item.source_units for item in out.follow_up_questions)


def test_stage_5_drops_ungrounded_generic_question_instead_of_defaulting_u1():
    units = [
        ConversationUnit(id="U1", text="Customers should reserve tables online."),
        ConversationUnit(id="U2", text="The design should feel clean and modern."),
    ]
    try:
        validate_stage_5_followups(
            {"follow_up_questions": ["What should we do next?"]},
            units,
        )
        assert False, "Expected ExtractionError for ungrounded question"
    except ExtractionError as exc:
        assert "ungrounded questions" in str(exc)


def test_semantic_verify_downgrades_hard_boundary_constraint_with_wrong_source():
    units = [
        ConversationUnit(id="U1", text="Customers should reserve tables online."),
        ConversationUnit(id="U2", text="The website should load quickly on mobile."),
    ]
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[],
        non_functional_requirements=[],
        constraints=[
            ConstraintItem(
                id="CON1",
                text="Online payment shall not be included in the initial release.",
                source_units=["U2"],
            )
        ],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=units,
    )
    verified, warnings = semantic_verify(spec, units)
    assert not verified.constraints
    assert any("hard project constraint" in q.text.lower() for q in verified.open_questions)
    assert any("low_semantic_overlap" in w for w in warnings)


def test_semantic_verify_repairs_constraint_sources_when_better_evidence_exists():
    units = [
        ConversationUnit(id="U1", text="Online payment is not part of the first release."),
        ConversationUnit(id="U2", text="Customers should reserve tables by phone."),
    ]
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[],
        non_functional_requirements=[],
        constraints=[
            ConstraintItem(
                id="CON1",
                text="Online payment shall not be included in the initial release.",
                source_units=["U2"],
            )
        ],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=units,
    )
    verified, warnings = semantic_verify(spec, units)
    assert len(verified.constraints) == 1
    assert verified.constraints[0].source_units == ["U1"]
    assert any("source_units_repaired_by_inference" in w for w in warnings)


def test_stage_1_relabels_quality_like_requirement_as_quality_expectation():
    units = [ConversationUnit(id="U1", text="The site should load quickly on mobile.")]
    out = validate_stage_1_candidates(
        {
            "candidates": [
                {
                    "id": "C1",
                    "kind": "possible_requirement",
                    "text": "The site should load quickly on mobile.",
                    "source_units": ["U1"],
                }
            ]
        },
        units,
    )
    assert out.candidates[0].kind == "possible_quality_expectation"


def test_stage_2_sanity_coerces_mobile_performance_fr_to_nfr():
    units = [ConversationUnit(id="U1", text="The site should load quickly on mobile.")]
    stage1 = Stage1CandidatesOutput(
        candidates=[
            CandidateItem(
                id="C1",
                kind="possible_quality_expectation",
                text="The site should load quickly on mobile.",
                source_units=["U1"],
            )
        ]
    )
    out = validate_stage_2_classification(
        {
            "classified_candidates": [
                {
                    "id": "C1",
                    "final_type": "functional_requirement",
                    "reason": "Model marked as functional.",
                    "source_units": ["U1"],
                }
            ]
        },
        stage1,
        units,
    )
    assert out.classified_candidates[0].final_type == "non_functional_requirement"


def test_stage_2_sanity_coerces_clean_modern_fr_to_open_question():
    units = [ConversationUnit(id="U1", text="The design should feel clean and modern.")]
    stage1 = Stage1CandidatesOutput(
        candidates=[
            CandidateItem(
                id="C1",
                kind="possible_quality_expectation",
                text="The design should feel clean and modern.",
                source_units=["U1"],
            )
        ]
    )
    out = validate_stage_2_classification(
        {
            "classified_candidates": [
                {
                    "id": "C1",
                    "final_type": "functional_requirement",
                    "reason": "Model marked as functional.",
                    "source_units": ["U1"],
                }
            ]
        },
        stage1,
        units,
    )
    assert out.classified_candidates[0].final_type == "open_question"


def test_stage_2_sanity_keeps_true_capability_as_fr():
    units = [ConversationUnit(id="U1", text="Customers should be able to reserve tables online.")]
    stage1 = Stage1CandidatesOutput(
        candidates=[
            CandidateItem(
                id="C1",
                kind="possible_requirement",
                text="Customers should be able to reserve tables online.",
                source_units=["U1"],
            )
        ]
    )
    out = validate_stage_2_classification(
        {
            "classified_candidates": [
                {
                    "id": "C1",
                    "final_type": "functional_requirement",
                    "reason": "Model marked as functional.",
                    "source_units": ["U1"],
                }
            ]
        },
        stage1,
        units,
    )
    assert out.classified_candidates[0].final_type == "functional_requirement"


def test_stage_1_rejects_placeholder_echo_candidates():
    units = [ConversationUnit(id="U1", text="We need a cafe website.")]
    try:
        validate_stage_1_candidates(
            {
                "candidates": [
                    {
                        "id": "C1",
                        "kind": "possible_requirement",
                        "text": "short candidate description",
                        "source_units": ["U1"],
                    },
                    {
                        "id": "C2",
                        "kind": "possible_constraint",
                        "text": "explicit boundary or release limitation",
                        "source_units": ["U1"],
                    },
                ]
            },
            units,
        )
        assert False, "Expected ExtractionError for placeholder echo candidates"
    except ExtractionError as exc:
        assert "placeholder/example echo" in str(exc)


def test_stage_3_drops_items_not_grounded_in_stage_2_candidates():
    units = [
        ConversationUnit(id="U1", text="Customers should reserve tables online."),
        ConversationUnit(id="U2", text="The design should feel clean and modern."),
    ]
    out = validate_stage_3_rewriting(
        {
            "rewritten_items": [
                {
                    "id": "R1",
                    "type": "functional_requirement",
                    "text": "The system shall allow customers to reserve tables online.",
                    "source_units": ["U1"],
                },
                {
                    "id": "R2",
                    "type": "functional_requirement",
                    "text": "The system shall provide a clean and modern design.",
                    "source_units": ["U2"],
                },
            ]
        },
        units,
        authorized_rewrite_candidates=[
            {
                "id": "C1",
                "final_type": "functional_requirement",
                "text": "Customers should reserve tables online.",
                "source_units": ["U1"],
            }
        ],
    )
    assert len(out.rewritten_items) == 1
    assert out.rewritten_items[0].source_units == ["U1"]


def test_requirement_quality_enrichment_fills_missing_model_fields():
    units = [ConversationUnit(id="U1", text="Customers should reserve tables online.")]
    rewritten = Stage3RewrittenOutput(
        rewritten_items=[
            RewrittenItem(
                id="R1",
                type="functional_requirement",
                text="The system shall allow customers to reserve tables online.",
                source_units=["U1"],
            )
        ]
    )
    out = validate_requirement_quality_enrichment(
        {
            "enriched_items": [
                {
                    "id": "R1",
                    "type": "functional_requirement",
                    "text": "The system shall allow customers to reserve tables online.",
                    "source_units": ["U1"],
                    "quality_checks": {
                        "is_atomic": True,
                        "is_testable": True,
                        "has_clear_actor": True,
                        "has_traceable_evidence": True,
                        "ambiguity_risk": "low",
                    },
                }
            ]
        },
        units,
        rewritten,
    )
    assert len(out.enriched_items) == 1
    assert out.enriched_items[0].evidence_spans == ["Customers should reserve tables online."]
    assert out.enriched_items[0].acceptance_criteria
    assert out.enriched_items[0].quality_checks.has_traceable_evidence is True


def test_stage_1_fallback_marks_mobile_performance_as_quality_expectation():
    units = [
        ConversationUnit(id="U1", text="The website should load quickly on mobile devices."),
        ConversationUnit(id="U2", text="Customers should reserve tables online."),
    ]
    out = build_stage_1_fallback_candidates(units)
    quality_items = [c for c in out.candidates if c.kind == "possible_quality_expectation"]
    assert any("load quickly on mobile" in item.text.lower() for item in quality_items)
