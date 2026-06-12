from __future__ import annotations

from typing import Iterable

from app.schemas import ConstraintItem, NoteItem, QuestionItem, RequirementItem, SpecOutput


def _format_traceable_item(item: RequirementItem | ConstraintItem) -> list[str]:
    sources = ", ".join(item.source_units) if item.source_units else "-"
    lines = [f"- **{item.id}**: {item.text} _(source_units: {sources})_"]
    if item.evidence_spans:
        evidence = " | ".join(item.evidence_spans)
        lines.append(f"  - Evidence: {evidence}")
    if item.acceptance_criteria:
        lines.append("  - Acceptance criteria:")
        lines.extend([f"    - {criterion}" for criterion in item.acceptance_criteria])
    return lines


def _format_requirement_list(items: Iterable[RequirementItem]) -> list[str]:
    lines: list[str] = []
    for item in items:
        lines.extend(_format_traceable_item(item))
    if not lines:
        lines.append("- None")
    return lines


def _format_constraint_list(items: Iterable[ConstraintItem]) -> list[str]:
    lines: list[str] = []
    for item in items:
        lines.extend(_format_traceable_item(item))
    if not lines:
        lines.append("- None")
    return lines


def _format_question_list(items: Iterable[QuestionItem | NoteItem]) -> list[str]:
    lines: list[str] = []
    for item in items:
        sources = ", ".join(item.source_units) if item.source_units else "-"
        lines.append(f"- {item.text} _(source_units: {sources})_")
    if not lines:
        lines.append("- None")
    return lines


def format_spec_markdown(spec: SpecOutput) -> str:
    lines: list[str] = ["# Software Requirements Draft", ""]

    lines.extend(["## Project Summary", spec.project_summary, ""])
    lines.extend(["## Functional Requirements", *_format_requirement_list(spec.functional_requirements), ""])
    lines.extend(
        [
            "## Non-functional Requirements",
            *_format_requirement_list(spec.non_functional_requirements),
            "",
        ]
    )
    lines.extend(["## Constraints", *_format_constraint_list(spec.constraints), ""])
    lines.extend(["## Open Questions", *_format_question_list(spec.open_questions), ""])
    lines.extend(["## Follow-up Questions", *_format_question_list(spec.follow_up_questions), ""])
    lines.extend(["## Notes", *_format_question_list(spec.notes), ""])
    return "\n".join(lines).rstrip() + "\n"
