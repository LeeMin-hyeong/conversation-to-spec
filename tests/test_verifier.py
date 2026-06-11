import json

from app.pipeline import ConversationToSpecPipeline
from app.prompt_builder import build_single_shot_spec_prompt, load_prompt_config
from app.schemas import (
    ConstraintItem,
    ConversationUnit,
    RequirementItem,
    RequirementVerification,
    SpecOutput,
)
from app.verifier import (
    SpecVerifier,
    heuristic_verification,
    needs_selective_repair,
    numeric_threshold_details,
    source_relevance_score,
    unsupported_numeric_details,
)
from tests.fakes import FakeSingleShotRunner


def _units() -> list[ConversationUnit]:
    return [
        ConversationUnit(id="U1", text="Customers should reserve tables online."),
        ConversationUnit(id="U2", text="The design should feel modern."),
    ]


def test_few_shot_prompt_contains_compact_example_and_trace_fields():
    prompt = build_single_shot_spec_prompt(
        _units(),
        load_prompt_config(),
        prompt_style="few_shot",
    )
    assert "Few-shot example" in prompt
    assert "source_unit_decisions" in prompt
    assert "functional_requirement" in prompt
    assert "repeat the same source_unit" in prompt
    assert "Do not output an atomic_decisions field" in prompt
    assert "Do not generate final functional_requirements" in prompt
    assert "EX_ACTOR_A" in prompt
    assert "Never output EX_* tokens" in prompt
    assert "book tables" not in prompt


def test_requirement_schema_accepts_verification_fields():
    item = RequirementItem(
        id="FR1",
        text="The system shall allow customers to reserve tables online.",
        source_units=["U1"],
        verification=RequirementVerification(
            source_relevance_score=0.91,
            verdict="SUPPORTED",
            confidence=0.88,
            warnings=[],
        ),
    )
    assert item.verification.verdict == "SUPPORTED"


def test_source_relevance_fallback_scoring():
    score = source_relevance_score(
        "The system shall allow customers to reserve tables online.",
        "Customers should reserve tables online.",
    )
    assert score > 0.4


def test_heuristic_verdict_assignment_supported_and_unsupported():
    supported = RequirementItem(
        id="FR1",
        text="The system shall allow customers to reserve tables online.",
        source_units=["U1"],
        evidence_spans=["Customers should reserve tables online."],
        acceptance_criteria=["Given a customer, When they reserve, Then it is saved."],
    )
    supported_verification = heuristic_verification(
        supported,
        {"U1": "Customers should reserve tables online."},
    )
    assert supported_verification.verdict == "SUPPORTED"

    unsupported = RequirementItem(
        id="FR2",
        text="The system shall support blockchain settlement.",
        source_units=["U9"],
        evidence_spans=[],
        acceptance_criteria=[],
    )
    unsupported_verification = heuristic_verification(
        unsupported,
        {"U1": "Customers should reserve tables online."},
    )
    assert unsupported_verification.verdict == "UNSUPPORTED"
    assert "source_units_not_found" in unsupported_verification.warnings


def test_heuristic_flags_claim_terms_missing_from_evidence():
    item = RequirementItem(
        id="NFR1",
        text="Rooms should have unique identifiers.",
        source_units=["U1", "U2"],
        evidence_spans=[
            "Staff should be able to block rooms for maintenance.",
            "Students should be able to find rooms by time and capacity.",
        ],
        acceptance_criteria=["Given rooms are listed, When records are checked, Then identifiers exist."],
    )
    verification = heuristic_verification(
        item,
        {
            "U1": "Staff should be able to block rooms for maintenance.",
            "U2": "Students should be able to find rooms by time and capacity.",
        },
    )
    assert verification.verdict == "UNSUPPORTED"
    assert any(warning.startswith("unsupported_claim_terms:") for warning in verification.warnings)


def test_heuristic_verifier_prefers_atomic_evidence_for_contradiction_check():
    item = RequirementItem(
        id="FR1",
        text="The system shall allow staff to see who changed an application status and when.",
        source_units=["U1"],
        evidence_spans=[
            "For audit reasons, staff should be able to see who changed an application status and when"
        ],
        acceptance_criteria=[
            "Given staff review an application, When status history is opened, Then the system shall show who changed the status and when."
        ],
    )
    verification = heuristic_verification(
        item,
        {
            "U1": (
                "For audit reasons, staff should be able to see who changed an application status "
                "and when, although detailed legal audit exports can wait until phase two."
            )
        },
    )

    assert verification.verdict == "SUPPORTED"


def test_deferred_release_terms_match_wait_until_evidence():
    item = ConstraintItem(
        id="CON1",
        text="Detailed legal audit exports shall be deferred to a later release.",
        source_units=["U1"],
        evidence_spans=["detailed legal audit exports can wait until phase two"],
        acceptance_criteria=[
            "Given the first release scope is reviewed, When scope decisions are checked, Then detailed legal audit exports are deferred."
        ],
    )
    verification = heuristic_verification(
        item,
        {"U1": "Staff need audit history, although detailed legal audit exports can wait until phase two."},
    )

    assert verification.verdict == "SUPPORTED"
    assert not any(warning.startswith("unsupported_claim_terms") for warning in verification.warnings)


def test_numeric_threshold_details_are_checked_against_evidence():
    assert numeric_threshold_details("Then the page shall load within 2 seconds.") == ["2 second"]
    assert unsupported_numeric_details(
        "Then the page shall load within 2 seconds.",
        "The site should load quickly on phones.",
    ) == ["2 second"]
    assert unsupported_numeric_details(
        "Then the page shall load within 2 seconds.",
        "The site should load within 2 seconds on phones.",
    ) == []


def test_heuristic_treats_not_part_of_launch_as_exclusion_evidence():
    item = ConstraintItem(
        id="CON1",
        text="Online rent payment shall be excluded from the launch scope.",
        source_units=["U1"],
        evidence_spans=["Online rent payment may be added later, but it should not be part of launch."],
        acceptance_criteria=[
            "Given the first release scope is reviewed, When scope decisions are checked, "
            "Then the constraint is satisfied: Online rent payment shall be excluded from the launch scope."
        ],
    )

    verification = heuristic_verification(
        item,
        {"U1": "Online rent payment may be added later, but it should not be part of launch."},
    )

    assert verification.verdict == "SUPPORTED"
    assert not any(warning.startswith("unsupported_claim_terms") for warning in verification.warnings)


def test_heuristic_marks_ungrounded_numeric_acceptance_as_partial():
    item = RequirementItem(
        id="NFR1",
        text="The system shall load quickly on mobile.",
        source_units=["U1"],
        evidence_spans=["The site should load quickly on phones."],
        acceptance_criteria=[
            "Given a mobile user opens the site, When the page loads, Then it shall load within 2 seconds."
        ],
    )
    verification = heuristic_verification(
        item,
        {"U1": "The site should load quickly on phones."},
    )
    assert verification.verdict == "PARTIALLY_SUPPORTED"
    assert any(warning.startswith("unsupported_numeric_detail:2 second") for warning in verification.warnings)


def test_selective_repair_trigger_logic():
    item = RequirementItem(
        id="FR1",
        text="The system shall support unsupported behavior.",
        source_units=["U1"],
        evidence_spans=["Something else."],
        acceptance_criteria=["Given X, When Y, Then Z."],
        verification=RequirementVerification(verdict="UNSUPPORTED"),
    )
    assert needs_selective_repair(item) is True


def test_selective_repair_trigger_for_ungrounded_numeric_detail():
    item = RequirementItem(
        id="NFR1",
        text="The system shall load quickly on mobile.",
        source_units=["U1"],
        evidence_spans=["The site should load quickly on phones."],
        acceptance_criteria=[
            "Given a mobile user opens the site, When the page loads, Then it shall load within 2 seconds."
        ],
        verification=RequirementVerification(
            verdict="PARTIALLY_SUPPORTED",
            warnings=["unsupported_numeric_detail:2 second"],
        ),
    )
    assert needs_selective_repair(item) is True


def test_spec_verifier_repairs_unsupported_requirement_to_open_question():
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[
            RequirementItem(
                id="FR1",
                text="The system shall support blockchain settlement.",
                source_units=["U1"],
                evidence_spans=["Customers should reserve tables online."],
                acceptance_criteria=["Given X, When Y, Then Z."],
            )
        ],
        conversation_units=_units(),
    )
    result = SpecVerifier().run(
        spec,
        _units(),
        verify_mode="heuristic",
        repair_on_fail=True,
    )
    assert not result.spec.functional_requirements
    assert result.spec.open_questions
    assert result.report["summary"]["repair_trigger_count"] == 1
    assert result.report["summary"]["repair_success_count"] == 1


def test_spec_verifier_removes_duplicate_questions_for_repaired_requirement():
    spec = SpecOutput(
        project_summary="Summary.",
        non_functional_requirements=[
            RequirementItem(
                id="NFR1",
                text="Rooms should have unique identifiers.",
                source_units=["U1"],
                evidence_spans=["Students should find rooms by time."],
                acceptance_criteria=["Given rooms exist, When records are checked, Then identifiers exist."],
            )
        ],
        open_questions=[
            {"text": "How do we ensure that all rooms have unique identifiers?", "source_units": ["U1"]},
            {"text": "Which rooms are searchable?", "source_units": ["U1"]},
        ],
        conversation_units=[ConversationUnit(id="U1", text="Students should find rooms by time.")],
    )
    result = SpecVerifier().run(
        spec,
        [ConversationUnit(id="U1", text="Students should find rooms by time.")],
        verify_mode="heuristic",
        repair_on_fail=True,
    )
    question_texts = [item.text for item in result.spec.open_questions]
    assert "How do we ensure that all rooms have unique identifiers?" not in question_texts
    assert any("Please confirm or revise unsupported requirement NFR1" in text for text in question_texts)


def test_spec_verifier_uses_llm_for_language_repair_only_when_flagged():
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[
            RequirementItem(
                id="FR1",
                text="The system shall support this capability: Staff need a dashboard to manage requests.",
                source_units=["U1"],
                evidence_spans=["Staff need a dashboard to manage requests."],
                acceptance_criteria=[
                    "Given the requirement is in scope, When the related workflow is performed, Then the system shall support this capability."
                ],
            )
        ],
        non_functional_requirements=[],
        constraints=[],
        open_questions=[],
        follow_up_questions=[],
        notes=[],
        conversation_units=[
            ConversationUnit(id="U1", text="Staff need a dashboard to manage requests.")
        ],
    )
    runner = FakeSingleShotRunner()

    result = SpecVerifier(runner=runner).run(
        spec,
        spec.conversation_units,
        verify_mode="heuristic",
        repair_on_fail=True,
    )

    assert result.num_llm_calls == 1
    assert result.spec.functional_requirements[0].text == (
        "The system shall provide a dashboard for staff to manage requests."
    )
    assert result.spec.functional_requirements[0].acceptance_criteria[0].startswith("Given staff open")
    assert result.report["summary"]["repair_success_count"] == 1


def test_spec_verifier_repairs_numeric_detail_to_open_question():
    spec = SpecOutput(
        project_summary="Summary.",
        non_functional_requirements=[
            RequirementItem(
                id="NFR1",
                text="The system shall load quickly on mobile.",
                source_units=["U1"],
                evidence_spans=["The site should load quickly on phones."],
                acceptance_criteria=[
                    "Given a mobile user opens the site, When the page loads, Then it shall load within 2 seconds."
                ],
            )
        ],
        conversation_units=[
            ConversationUnit(id="U1", text="The site should load quickly on phones."),
        ],
    )
    result = SpecVerifier().run(
        spec,
        [ConversationUnit(id="U1", text="The site should load quickly on phones.")],
        verify_mode="heuristic",
        repair_on_fail=True,
    )
    repaired = result.spec.non_functional_requirements[0]
    assert "2 seconds" not in " ".join(repaired.acceptance_criteria)
    assert repaired.verification.verdict in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
    assert any("What measurable target should define NFR1" in q.text for q in result.spec.open_questions)
    assert result.report["summary"]["repair_trigger_count"] == 1


class _FakeMiniCheckScorer:
    def score(self, *, document: str, claim: str) -> tuple[int, float]:
        assert "Customers should reserve tables online." in document
        assert "reserve tables" in claim
        return 1, 0.87


class _FakeAtomicMiniCheckScorer:
    def score(self, *, document: str, claim: str) -> tuple[int, float]:
        assert "see who changed an application status" in document
        assert "phase two" not in document
        assert "see who changed an application status" in claim
        return 1, 0.91


def test_spec_verifier_supports_minicheck_mode_with_injected_scorer():
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[
            RequirementItem(
                id="FR1",
                text="The system shall allow customers to reserve tables online.",
                source_units=["U1"],
                evidence_spans=["Customers should reserve tables online."],
                acceptance_criteria=["Given a customer, When they reserve, Then it is saved."],
            )
        ],
        conversation_units=_units(),
    )
    result = SpecVerifier(minicheck_scorer=_FakeMiniCheckScorer()).run(
        spec,
        _units(),
        verify_mode="minicheck",
    )
    assert result.spec.functional_requirements[0].verification.verdict == "SUPPORTED"
    assert result.spec.functional_requirements[0].verification.confidence == 0.87


def test_minicheck_verifier_uses_atomic_evidence_for_contradiction_check():
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[
            RequirementItem(
                id="FR1",
                text="The system shall allow staff to see who changed an application status and when.",
                source_units=["U1"],
                evidence_spans=[
                    "For audit reasons, staff should be able to see who changed an application status and when"
                ],
                acceptance_criteria=[
                    "Given staff opens the relevant page, When the page loads, Then the system shall display who changed an application status and when."
                ],
            )
        ],
        conversation_units=[
            ConversationUnit(
                id="U1",
                text=(
                    "For audit reasons, staff should be able to see who changed an application status "
                    "and when, although detailed legal audit exports can wait until phase two."
                ),
            )
        ],
    )

    result = SpecVerifier(minicheck_scorer=_FakeAtomicMiniCheckScorer()).run(
        spec,
        spec.conversation_units,
        verify_mode="minicheck",
    )

    verification = result.spec.functional_requirements[0].verification
    assert verification.verdict == "SUPPORTED"
    assert verification.confidence == 0.91
    assert "rule_detected_scope_contradiction" not in verification.warnings


def test_fake_pipeline_writes_verification_report(tmp_path):
    pipeline = ConversationToSpecPipeline(
        runner=FakeSingleShotRunner(),
        prompt_config=load_prompt_config(),
        generation_config={"max_retries": 0},
        prompt_style="few_shot",
        verify_mode="heuristic",
        repair_on_fail=True,
    )
    run = pipeline.run_text(
        "Customers should reserve tables online.",
        output_dir=tmp_path,
    )
    assert run.success is True
    assert (tmp_path / "verification_report.json").exists()
    assert (tmp_path / "verification_report.md").exists()
    report = json.loads((tmp_path / "verification_report.json").read_text(encoding="utf-8"))
    assert "requirements" in report
    assert "summary" in report
