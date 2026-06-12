from app.quality import default_acceptance_criteria, validate_requirement_quality_items
from app.schemas import (
    ConversationUnit,
    RequirementItem,
    RequirementQualityChecks,
    SpecOutput,
)


def test_requirement_schema_accepts_minimal_and_enriched_items():
    minimal = RequirementItem(
        id="FR1",
        text="The system shall support booking.",
        source_units=["U1"],
    )
    assert minimal.evidence_spans == []
    assert minimal.acceptance_criteria == []
    assert minimal.quality_checks.ambiguity_risk == "medium"

    enriched = RequirementItem(
        id="FR2",
        text="The system shall allow customers to reserve tables.",
        source_units=["U1"],
        evidence_spans=["Customers should reserve tables online."],
        acceptance_criteria=[
            "Given a customer, When they reserve a table, Then the reservation is saved."
        ],
        quality_checks=RequirementQualityChecks(
            is_atomic=True,
            is_testable=True,
            has_clear_actor=True,
            has_traceable_evidence=True,
            ambiguity_risk="low",
        ),
    )
    spec = SpecOutput(
        project_summary="Summary.",
        functional_requirements=[minimal, enriched],
        conversation_units=[ConversationUnit(id="U1", text="Customers should reserve tables online.")],
    )
    assert spec.functional_requirements[1].quality_checks.has_traceable_evidence is True


def test_quality_validator_emits_traceability_and_testability_warnings():
    units = [ConversationUnit(id="U1", text="The site should load quickly on mobile.")]
    item = RequirementItem(
        id="NFR1",
        text="The system should be fast and reliable.",
        source_units=["U9"],
        evidence_spans=[],
        acceptance_criteria=[""],
        quality_checks=RequirementQualityChecks(
            is_atomic=True,
            is_testable=True,
            has_clear_actor=True,
            has_traceable_evidence=True,
            ambiguity_risk="medium",
        ),
    )
    warnings = validate_requirement_quality_items(
        [item],
        units,
        category_label="non_functional_requirements",
    )
    joined = "\n".join(warnings)
    assert "unknown source_units" in joined
    assert "marked traceable but has no evidence spans" in joined
    assert "has no acceptance criteria" in joined
    assert "empty acceptance criterion" in joined
    assert "vague quality wording" in joined


def test_default_acceptance_criteria_uses_readable_domain_neutral_actions():
    menu = default_acceptance_criteria(
        "The system shall allow customers to see today's menu and opening hours."
    )[0]
    reservation = default_acceptance_criteria(
        "The system shall allow customers to reserve tables online for specific time slots."
    )[0]
    admin = default_acceptance_criteria(
        "The system shall allow staff to update menu items and prices from an admin page."
    )[0]

    assert "shall display today's menu and opening hours" in menu
    assert "save the reservation with the selected time slot" in reservation
    assert "store the updated menu items and prices" in admin
    assert "the see" not in menu.lower()
    assert "record the reserve" not in reservation.lower()


def test_default_acceptance_criteria_reuses_domain_actor_for_notifications():
    resident_update = default_acceptance_criteria(
        "The system shall allow residents to receive SMS or email updates when a technician is assigned."
    )[0]
    staff_access = default_acceptance_criteria(
        "The system shall allow authorized office staff to access student health notes."
    )[0]
    alert = default_acceptance_criteria(
        "The system shall send an email alert when an item is close to running out or near expiration."
    )[0]
    reminder = default_acceptance_criteria(
        "The system shall send reminder emails to parents who have not submitted a form."
    )[0]

    assert "to the resident" in resident_update
    assert "SMS or email update" in resident_update
    assert "customer" not in resident_update.lower()
    assert "Given authorized office staff starts" in staff_access
    assert "Given a authorized" not in staff_access
    assert "Given an item is close to running out or near expiration" in alert
    assert "parents who have not submitted a form are eligible" in reminder
