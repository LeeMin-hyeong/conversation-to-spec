from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ConversationUnit(BaseModel):
    id: str
    text: str


class RequirementItem(BaseModel):
    id: str
    text: str
    source_units: List[str] = Field(default_factory=list)


class ConstraintItem(BaseModel):
    id: str
    text: str
    source_units: List[str] = Field(default_factory=list)


class QuestionItem(BaseModel):
    text: str
    source_units: List[str] = Field(default_factory=list)


class NoteItem(BaseModel):
    text: str
    source_units: List[str] = Field(default_factory=list)


class SpecOutput(BaseModel):
    project_summary: str
    functional_requirements: List[RequirementItem] = Field(default_factory=list)
    non_functional_requirements: List[RequirementItem] = Field(default_factory=list)
    constraints: List[ConstraintItem] = Field(default_factory=list)
    open_questions: List[QuestionItem] = Field(default_factory=list)
    follow_up_questions: List[QuestionItem] = Field(default_factory=list)
    notes: List[NoteItem] = Field(default_factory=list)
    conversation_units: List[ConversationUnit] = Field(default_factory=list)
    verification_warnings: List[str] = Field(default_factory=list)


class CandidateItem(BaseModel):
    id: str
    kind: str
    text: str
    source_units: List[str] = Field(default_factory=list)


class Stage1CandidatesOutput(BaseModel):
    candidates: List[CandidateItem] = Field(default_factory=list)


class ClassifiedCandidateItem(BaseModel):
    id: str
    final_type: str
    reason: str
    source_units: List[str] = Field(default_factory=list)


class Stage2ClassifiedOutput(BaseModel):
    classified_candidates: List[ClassifiedCandidateItem] = Field(default_factory=list)


class RewrittenItem(BaseModel):
    id: str
    type: str
    text: str
    source_units: List[str] = Field(default_factory=list)


class Stage3RewrittenOutput(BaseModel):
    rewritten_items: List[RewrittenItem] = Field(default_factory=list)


class Stage4OpenQuestionsOutput(BaseModel):
    open_questions: List[QuestionItem] = Field(default_factory=list)


class Stage5FollowUpOutput(BaseModel):
    follow_up_questions: List[QuestionItem] = Field(default_factory=list)


class Stage6SummaryOutput(BaseModel):
    project_summary: str


# Backward-compatible aliases for old stage numbering.
class Stage4FollowUpOutput(Stage5FollowUpOutput):
    pass


class Stage5SummaryOutput(Stage6SummaryOutput):
    pass
