from app.extractor import (
    build_stage_1_fallback_candidates,
    ExtractionError,
    extract_spec_output,
    semantic_verify,
    validate_stage_1_candidates,
    validate_stage_2_classification,
    validate_stage_3_rewriting,
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


def test_semantic_verify_downgrades_future_scope_constraint_to_note():
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
    assert not verified.constraints
    assert any("Future-scope boundary candidate" in note.text for note in verified.notes)
    assert any("future_scope_without_explicit_boundary" in w for w in warnings)


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


def test_stage_1_fallback_marks_mobile_performance_as_quality_expectation():
    units = [
        ConversationUnit(id="U1", text="The website should load quickly on mobile devices."),
        ConversationUnit(id="U2", text="Customers should reserve tables online."),
    ]
    out = build_stage_1_fallback_candidates(units)
    quality_items = [c for c in out.candidates if c.kind == "possible_quality_expectation"]
    assert any("load quickly on mobile" in item.text.lower() for item in quality_items)
