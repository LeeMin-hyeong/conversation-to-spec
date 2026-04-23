from app.formatter import format_spec_markdown
from app.schemas import ConstraintItem, ConversationUnit, NoteItem, QuestionItem, RequirementItem, SpecOutput


def test_markdown_contains_required_sections():
    spec = SpecOutput(
        project_summary="Project summary text.",
        functional_requirements=[
            RequirementItem(id="FR1", text="The system shall do X.", source_units=["U1"])
        ],
        non_functional_requirements=[
            RequirementItem(id="NFR1", text="The system should do Y.", source_units=["U2"])
        ],
        constraints=[
            ConstraintItem(id="CON1", text="Online payment shall not be included in v1.", source_units=["U4"])
        ],
        open_questions=[QuestionItem(text="What is the SLA?", source_units=["U3"])],
        follow_up_questions=[
            QuestionItem(text="Can you define SLA target?", source_units=["U3"])
        ],
        notes=[NoteItem(text="Future scope item.", source_units=["U4"])],
        conversation_units=[ConversationUnit(id="U1", text="raw text")],
    )
    md = format_spec_markdown(spec)
    assert "## Project Summary" in md
    assert "## Functional Requirements" in md
    assert "## Non-functional Requirements" in md
    assert "## Constraints" in md
    assert "## Open Questions" in md
    assert "## Follow-up Questions" in md
    assert "## Notes" in md
    assert "Future scope item." in md
