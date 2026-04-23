from __future__ import annotations

from typing import Iterable

from app.schemas import ConstraintItem, NoteItem, QuestionItem, RequirementItem, SpecOutput


def _format_requirement_list(items: Iterable[RequirementItem]) -> list[str]:
    lines: list[str] = []
    for item in items:
        sources = ", ".join(item.source_units) if item.source_units else "-"
        lines.append(f"- **{item.id}**: {item.text} _(source_units: {sources})_")
    if not lines:
        lines.append("- None")
    return lines


def _format_constraint_list(items: Iterable[ConstraintItem]) -> list[str]:
    lines: list[str] = []
    for item in items:
        sources = ", ".join(item.source_units) if item.source_units else "-"
        lines.append(f"- **{item.id}**: {item.text} _(source_units: {sources})_")
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
    if spec.verification_warnings:
        lines.extend(["## Verification Warnings"])
        lines.extend([f"- {warning}" for warning in spec.verification_warnings])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
